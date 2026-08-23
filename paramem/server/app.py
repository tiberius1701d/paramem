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
import functools
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from paramem.adapters.registry_binding import TierBinding
    from paramem.server.user_tokens import UserTokenStore

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Migration / backup imports at module level so tests can patch them.
from paramem.backup.backup import enforce_disk_cap, read_bundle_manifest, write_bundle
from paramem.backup.backup import write as backup_write
from paramem.backup.types import ArtifactKind, DiskCapExceeded
from paramem.cloud.providers import get_cloud_agent
from paramem.graph.extractor import ExtractionFailed
from paramem.graph.name_extraction import extract_name_via_llm
from paramem.graph.phase_trace import extraction_trace
from paramem.graph.prompts import prompt_overrides
from paramem.models.loader import (
    active_adapter_name,
    base_model_inference,
    has_prior_trained_weights,
    load_base_model,
    switch_adapter,
    unload_model,
)
from paramem.server import calibrate as calibrate_module
from paramem.server.active_store_migration import migrate
from paramem.server.background_trainer import BackgroundTrainer
from paramem.server.config import (
    DEFAULT_SERVER_CONFIG_PATH,
    TTSConfig,
    TTSVoiceConfig,
    default_data_dir,
    load_server_config,
)
from paramem.server.config_store_validator import ConfigStoreMismatch, check_config_against_store
from paramem.server.consolidation import (
    classify_pending_sessions,
    get_or_create_consolidation_loop,
    retire_unattributable_sessions,
)
from paramem.server.consolidation_action import ConsolidationAction, consolidation_content_gate
from paramem.server.ha_graph import HAEntityGraph
from paramem.server.incidents import (
    ack_incident,
    read_incidents,
    record_incident,
    resolve_incident,
    resolve_incidents_by_type,
)
from paramem.server.inference import (
    ChatResult,
    answer_via_cloud,
    handle_chat,
)
from paramem.server.router import QueryRouter
from paramem.server.run_status import read_last_runs, record_last_run
from paramem.server.sanitizer import is_self_referential
from paramem.server.session_buffer import SessionBuffer  # "session" here = conversation
from paramem.server.speaker import resolve_speaker_tokens
from paramem.server.tools.ha_client import HAClient
from paramem.server.trial_state import (
    TrialMarker,
    clear_trial_marker,
    read_trial_marker,
    trial_active,
    write_trial_marker,
)
from paramem.server.voice_pipeline import process_utterance
from paramem.server.vram_predict import predict_base_bytes
from paramem.server.vram_validator import (
    assess_topology,
    check_post_load_budget,
    estimate_stt_bytes,
    estimate_tts_bytes,
    format_baseline_fit,
)
from paramem.training.consolidation import (
    ActiveKeyHydrationFailure,
    PendingRelations,
    RecallGateRejected,
    enrichment_signal,
    interim_outcome_label,
)
from paramem.training.stage_ledger import data_state_dir
from paramem.training.thermal_throttle import ThermalPolicy, wait_for_cooldown
from paramem.utils import systemctl
from paramem.utils.artifacts import (
    artifact_run_dir,
    calibration_run,
    on_calibration_result,
    run_stamp,
)
from paramem.utils.identity import canonical as _canonical
from paramem.utils.identity import is_speaker_id as _is_speaker_id
from paramem.utils.notify import SERVER_CLOUD_ONLY, notify_server
from paramem.utils.paths import find_project_root
from paramem.utils.tokens import check_ratio_drift
from paramem.utils.vram_guard import (
    VramExhausted,
    apply_process_cap,
    check_vram_headroom,
    is_fatal_cuda_fault,
    safe_empty_cache,
    vram_measure,
    vram_scope,
)

logger = logging.getLogger(__name__)


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
    "cloud_agent": None,
    "cloud_providers": {},
    "ha_client": None,
    "consolidation_loop": None,
    "memory_store": None,
    # The latest calibration run submitted through /calibrate/* and, once
    # its terminal fired, its outcome — see StatusResponse.calibration_run.
    # Written only on status == "started_calibration" (_submit_calibration_run);
    # completed by _run_calibration_sync's own terminal.  One slot, not a
    # ring — reset on every process start.
    "calibration_run": None,
    # True once the store-preload step (_hydrate_memory_store_in_place, via
    # _build_store_contents) has cleanly completed for the CURRENT store
    # object — where completion includes the boot fill exactly when
    # inference.preload_cache=True and the venue can serve
    # (source.train_venue_deferred's predicate).  A train-venue pass with no
    # model resident leaves this False so the gate re-attempts on the next
    # act with a model (see _build_runtime_components's re-probe gate). A
    # shortfall in the fill itself is NOT a completeness failure — it is
    # pure telemetry, recorded as a preload_recall_incomplete incident by
    # _build_store_contents and surfaced via the generic incident attention
    # collector; it never clears this flag.  Never invalidated by a GPU
    # release — a cloud-only deferral does not invalidate the mirror.
    "store_preload_complete": False,
    # Set by _enter_store_quarantine when the boot/lift store step
    # (_hydrate_memory_store_in_place) cannot publish a fresh MemoryStore —
    # {"cause": {"exception_type", "message"}, "quarantined_at"}. ``None``
    # when the store is healthy. Surfaced verbatim on StatusResponse.store_quarantined.
    "store_quarantine": None,
    "consolidating": False,
    "last_consolidation": None,
    # NOTE: last_consolidation_error and last_consolidation_result are DERIVED
    # from the durable incident store and run_status registry at /status build
    # time (see _derive_consolidation_status_fields).  No RAM snapshot survives.
    "background_trainer": None,
    "reclaim_task": None,
    "config_path": None,
    "config_drift_task": None,
    # Base-swap orchestration task handle, shared by every launch site (the
    # lifespan's own crash-recovery resume, the /gpu/acquire deferred
    # Phase-B re-launch, and /migration/confirm's fresh Phase-A launch —
    # every ``asyncio.create_task(_run_base_swap_orchestration(...))`` call
    # site stores its handle here). Cleared by its own done callback once it
    # completes; awaited by the boot-completion task before any catch-up
    # work runs, and cancelled at shutdown alongside its task siblings. Reset
    # to None at every lifespan start (see the lifespan slot-hygiene block)
    # so a second lifespan in one process never awaits a stale handle.
    "base_swap_task": None,
    # Boot-completion catch-up task (timer reconcile + backup/consolidation
    # catch-up dispatch) — see _run_boot_completion_tasks. Cancelled at
    # shutdown and cleared so a repeated lifespan (TestClient) starts clean.
    "boot_completion_task": None,
    "mode": "local",  # "local" or "cloud-only"
    # "explicit", "training", "gpu_conflict", "cuda_fault_persistent",
    # "insufficient_vram", "reload_failed", "apply_failed", "config_refused",
    # "released", "live_reload", or None
    "cloud_only_reason": None,
    "cloud_only_startup": False,  # set by --cloud-only CLI flag before app start
    # (conversation_id, notice_kind) pairs already announced over the relay
    # path — notice_kind is "degraded" (_DEGRADED_SERVING_NOTICE) or
    # "speakerless" (_SPEAKERLESS_RELAY_NOTICE), see _notice_once.  Keyed by
    # the pair (not conversation_id alone) so a conversation that already
    # announced one kind still announces the OTHER kind once, the first
    # time it applies.
    "relay_notice_conversations": set(),
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
    # Set to True when the infrastructure integrity check
    # (verify_infrastructure_integrity, inside _preload_memory_store) finds
    # a failure.  Downstream migration must not run against a degraded
    # store (it would vacuously complete with no-op relocations).
    # _preload_memory_store runs on every boot AND every in-process config
    # apply/reload — its integrity-check success branch clears this back to
    # False, so a corrupt registry the operator restores and then re-applies
    # config for is re-checked and un-stuck without a process restart.
    "integrity_check_failed": False,
    # Whether the daily age identity was loadable at boot.  Set in lifespan
    # alongside ``encryption``; used by ``GET /integrity`` and the boot
    # integrity gate to distinguish no-key from corruption failures.
    "daily_loadable": False,
    # Per-component VRAM ledger (bytes).  Populated at each component load;
    # cleared at each component unload.  Keys: "base", "stt", "tts".
    # Truthfulness invariant: every key must be cleared on the matching
    # unload/release path — see _release_base_model_in_process and
    # _set_voice_pipeline_profile('cpu').
    "vram_components": {},
}


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
    speaker_embedding: list[float] | None = None  # Voice embedding from STT
    route: str | None = None  # Force routing: "ha", "cloud", or None (auto)


class ChatResponse(BaseModel):
    text: str
    escalated: bool = False
    speaker: str | None = None
    follow_up: str | None = None  # Server-initiated follow-up (e.g. introduction)


class BackupBlock(BaseModel):
    """Backup subsystem state for /status.

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
    # "explicit", "training", "gpu_conflict", "cuda_fault_persistent",
    # "insufficient_vram", "reload_failed", "apply_failed", "config_refused",
    # "released", "live_reload", or None
    cloud_only_reason: str | None
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
    # Seconds until the next interim cadence boundary. None when refresh
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
    # Per-adapter manifest-provenance rows.
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
    #   trial_started_at : ISO-8601 UTC, or None when no trial (or terminal
    #                    base-swap report) is stashed.  Not gated on
    #                    state=="trial": after a base swap finishes,
    #                    _finish_base_swap resets state to "live" but leaves
    #                    started_at on the terminal report, so this can be
    #                    non-None while state=="live" — see TrialStash.
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
    # Migrate footer.
    server_started_at: str = ""
    # Document-ingest state.
    # Pending session counts split by source type. Populated from
    # session_buffer.get_summary()["per_source_type"].
    pending_documents: int = 0
    pending_transcripts: int = 0
    # Auto-reclaim error tracking. Populated when the in-process reclaim
    # loop fails a tick; cleared on next successful reclaim. None means the last
    # reclaim completed cleanly (or none has run yet).
    # Shape: {"at": <iso8601>, "error": <str>, "attempt_count": <int>}
    last_reclaim_error: dict | None = None
    # Device-wide VRAM (in-process torch.cuda.mem_get_info — authoritative on
    # WSL2 where nvidia-smi --query-gpu=memory.used returns 0).
    # None when CUDA is unavailable or mem_get_info fails.
    vram_used_mib: int | None = None
    vram_total_mib: int | None = None
    # Sum of the per-component ledger below (bytes → MiB), or None when no
    # component has been measured yet.
    vram_paramem_mib: int | None = None
    # Per-component VRAM usage in MiB.  Keys present only for components that
    # are currently resident ("base", "stt", "tts").  Empty when nothing is
    # loaded or before the first load.
    vram_components: dict[str, int] = {}
    # Seconds until the next scheduled tick on which _is_full_cycle_due would
    # evaluate to True.  None when cadence is disabled (manual-only), when no
    # interim dirs exist yet, or when full_period is manual-only.
    next_full_consolidation_seconds: int | None = None
    # Active key count per store tier.  Keys are raw tier names
    # (e.g. "episodic", "semantic", "procedural",
    # "episodic_interim_<stamp>").  The renderer sums interim tiers.
    tier_key_counts: dict[str, int] = {}
    # Oldest un-folded interim stamp ("YYYYMMDDTHHMM") or None when the
    # interim ring is empty.  Used by the renderer to show the deadline
    # math for the next full consolidation.
    oldest_interim_stamp: str | None = None
    # Memory-store quarantine state. ``None`` when the store is healthy
    # (the boot/lift store step published it — see
    # ``_hydrate_memory_store_in_place``). When set, the store failed its
    # invariant (a bookkeeping-completeness violation, an unverified
    # tier-registry binding, or any other non-CUDA hydration failure) and
    # ``_state["memory_store"]`` is unset — every other subsystem (model,
    # STT/TTS, HA, tri-path routing) keeps serving normally; only the
    # parametric-memory recall arm is out (HA / cloud / abstention still
    # answer chat). Shape: ``{"cause": {"exception_type", "message"},
    # "quarantined_at"}``. The four consolidation endpoints, the
    # ``/migration/confirm`` and ``/migration/accept`` trial doors,
    # ``POST /speaker/forget``, and ``POST /interim/discard`` refuse with
    # ``"store_quarantined"`` while this is set; admin endpoints (this one
    # included) keep serving.
    store_quarantined: dict | None = None
    # The latest calibration run submitted through /calibrate/* and,
    # once its terminal fired, its outcome — {run_id, action, route,
    # artifact_dir, started_at, outcome, finished_at}.  Read from the live
    # `_state["calibration_run"]` slot, falling back to the durable
    # op_type="calibration" run-status row (see
    # _derive_consolidation_status_fields) when the live slot has never
    # been written this process lifetime.  A client never resolves its own
    # run through this field — it resolves it through the artifact_dir its
    # own 200 returned; this is for operator visibility only.  None when no
    # calibration run has ever been submitted.
    calibration_run: dict | None = None


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
    """Response schema for every dispatch endpoint — consolidation and
    calibration.

    Every route returning this model is declared
    ``response_model_exclude_none=True``.  The four consolidation routes
    never set ``run_id``/``artifact_dir``, so their wire shape is exactly
    ``{"status": ..., "action": ...}`` on every outcome — unchanged by
    these two additive fields.

    Attributes
    ----------
    status:
        Terminal status of the dispatch: ``started*`` (the run was submitted),
        ``noop_*`` (nothing to do), or ``deferred_*`` (blocked, retry later).
    action:
        What the request actually resolved to — ``"full"`` (the interim slots
        collapsed into main memory), ``"interim"`` (recent conversations
        absorbed into a new interim slot), ``"reconcile"`` (main memory rebuilt
        from its own stored knowledge — the same interim-ring absorption as a
        full fold, with pending sessions left pending), ``"auto"``
        when the dispatch was refused before the schedule could resolve it,
        ``"calibrate"`` (an operator-supplied calibration artifact), or
        ``"calibrate_pending"`` (a calibration probe over the pending NAMED
        session set).  ``POST /scheduled-tick`` is the only REST door that
        requests ``AUTO`` (the boot-completion catch-up task also requests
        it, but in-process rather than through this response schema), so
        this is how its caller learns which of the two the deadline math
        resolved to; every other consolidation door echoes the action it
        asked for directly.
    run_id:
        The submitted CALIBRATION run's identity — its UTC
        ``%Y%m%dT%H%M%SZ`` stamp.  Present exactly when *status* is
        ``"started_calibration"``, and absent otherwise — including on
        every other ``started_*`` outcome (a fold, which writes no
        calibration artifacts) and ``"started_migration"`` (the arbitrator
        pre-empted this request with an armed store migration).  A
        dispatch that submitted no calibration run has no run to identify.
    artifact_dir:
        Where the run writes ``response.json`` and its hook artifacts.
        Present under the same condition as *run_id*.  A client resolves
        its own run through this value, never by polling for someone
        else's.
    """

    status: str
    action: str = "none"
    run_id: str | None = None
    artifact_dir: str | None = None


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
        Human-readable document title (filename stem); used for
        ``GET /status`` attribution.
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
    document_filename:
        Original filename of the document (e.g. ``"notes.md"``).  Stored
        alongside the original bytes and used to name the file in the
        retention archive.
    document_b64:
        Base64-encoded raw bytes of the original file.  Stored to disk as
        ``<doc_id>.origdoc`` and archived together with the chunk JSONLs on
        retirement.  Decoded size must not exceed 25 MiB; the server returns
        HTTP 400 (``{"error":"document_too_large"}``) if the limit is
        exceeded.
    """

    speaker_id: str
    sessions: list[IngestChunk]
    document_filename: str
    document_b64: str


_ORIGDOC_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB


class IngestSessionsResponse(BaseModel):
    """Response body for ``POST /ingest-sessions``.

    Attributes
    ----------
    queued:
        Session IDs appended to the ``SessionBuffer``
        (form ``<doc_id>-c<chunk_index:03d>``).
    total_chunks:
        Always equals ``len(request.sessions)``.
    doc_id:
        Document group identifier (``"doc-" + secrets.token_hex(4)``)
        shared by all chunk sessions from this request.
    rejected_unknown_speaker:
        ``True`` when the speaker_id is not in ``SpeakerStore``.
    rejected_no_speaker_id:
        ``True`` when ``speaker_id`` is an empty string.
    """

    queued: list[str]
    total_chunks: int
    doc_id: str = ""
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
        The speaker ID to forget (e.g. ``"speaker0"``).  Exact match.

    Note
    ----
    There is exactly one operation (a stale-mark that withholds the key
    from serving immediately and is retired at its owning tier's own next
    rebuild — see :func:`speaker_forget`'s docstring); it has no variant to
    select.
    Discarding an interim slot wholesale (rather than staling one speaker's
    keys within it) is a separate operation — ``POST /interim/discard``.
    Unrecognised fields in the request body are ignored (``extra="ignore"``),
    so a caller still sending a ``strategy`` field is unaffected.
    """

    speaker_id: str


class TierRestampOutcome(BaseModel):
    """One tier's outcome from an erase door's registry-mutation + rebind pass.

    Shared response item for ``POST /speaker/forget`` and
    ``POST /debug/erase-keys`` — both build their ``tiers`` list from
    :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`'s
    per-tier :class:`~paramem.memory.persistence.RestampResult` (via the
    shared :func:`_stale_mark_keys` sequence). The registry mutation for
    every named tier already landed on disk regardless of ``outcome`` —
    an erase door never refuses (see
    :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`'s
    own docstring) — this model reports only whether the tier's slot
    manifest was rebound to match.

    Attributes
    ----------
    tier:
        Tier (or interim slot) name.
    outcome:
        ``"rebound"`` — the mutation landed and, when the tier had
        anything to bind, its slot manifest now matches. ``"unbound"`` —
        the mutation landed but the tier's slot manifest could not be
        rebound because :func:`~paramem.memory.persistence.plan_restamp`
        found no legal target; see ``reason``. ``"rebind_failed"`` — the
        mutation landed but an ``OSError`` or
        :class:`~paramem.adapters.manifest.ManifestError` was raised while
        attempting the rebind itself (a transient I/O failure, not a
        planning refusal); ``reason`` carries the exception's message.
        Both ``"unbound"`` and ``"rebind_failed"`` tiers will fail to bind
        on the next boot/reload until a consolidation fold or registry
        restore repairs them — surfaced via a ``tier_registry_unverified``
        incident and an ERROR log line.
    slot:
        The rebound slot's path, only when ``outcome == "rebound"`` AND a
        slot was actually re-stamped (a tier with nothing to bind has no
        slot to report); ``None`` otherwise.
    reason:
        ``None`` when ``outcome == "rebound"``. When ``outcome ==
        "unbound"``, one of :data:`~paramem.memory.persistence.KEYS_WITHOUT_SLOT`,
        :data:`~paramem.memory.persistence.NO_PRE_WRITE_HASH`, or
        :data:`~paramem.memory.persistence.SLOT_ORPHANED`. When
        ``outcome == "rebind_failed"``, the caught exception's message.
    """

    tier: str
    outcome: str
    slot: "str | None" = None
    reason: "str | None" = None


class SpeakerForgetResponse(BaseModel):
    """Response body for ``POST /speaker/forget``.

    Attributes
    ----------
    removed_speaker:
        ``True`` when the speaker profile was found and removed from
        :class:`~paramem.server.speaker.SpeakerStore`.  ``False`` when the
        speaker ID was unknown to the store (no profile to delete).
    staled_keys:
        Indexed-memory keys withheld in their owning tier's
        :class:`~paramem.training.key_registry.KeyRegistry` — a marker that
        reserves the id and carries no fingerprint (the active simhash does
        not survive the transition), not a hard erase. Every named key is
        immediately unreachable for serving (excluded from
        ``list_active()``); its content and bookkeeping row leave with the
        rest of the key at its owning tier's own next rebuild, when the key
        is genuinely retired rather than merely withheld. A tier's rebuild
        is a full consolidation or ``POST /reconsolidate`` (both rebuild
        every main tier) or an interim cycle (rebuilds only the slot it
        mints) — a tier no consolidation reaches keeps its markers
        indefinitely. A key already withheld (or unknown) is reported here
        too — re-erasing it is idempotent, not an error.
    discarded_sessions:
        Pending conversation IDs that were found in the
        :class:`~paramem.server.session_buffer.SessionBuffer` attributed to
        the speaker and discarded (JSONL deleted, turns dropped).
    tiers:
        Per-tier :class:`TierRestampOutcome` for every tier the erase
        touched — the registry mutation always landed; this reports
        whether the tier's slot manifest was rebound to match.
    unbound_tiers:
        Tier names left unbound — every ``TierRestampOutcome.outcome !=
        "rebound"``, i.e. ``"unbound"`` or ``"rebind_failed"`` — a non-empty
        list means at least one affected tier will fail to bind on the
        next boot/reload until repaired; see ``GET /integrity`` and the
        ``tier_registry_unverified`` incident it emits for each.
    """

    removed_speaker: bool
    staled_keys: list[str]
    discarded_sessions: list[str]
    tiers: list[TierRestampOutcome]
    unbound_tiers: list[str]


# --- Interim discard schemas ---

# THE one place the unconfirmed-request status code is declared.
# Deliberately 409 — collapses every "this will not mutate now" answer (busy,
# cloud-only, trial-active, unconfirmed) into the same refusal band a client
# already has to branch on.  Handler, tests, and the DEPLOYMENT.md example all
# read it from here rather than hard-coding the literal.
_INTERIM_DISCARD_UNCONFIRMED_STATUS: int = 409


class InterimDiscardRequest(BaseModel):
    """Request body for ``POST /interim/discard``.

    Attributes
    ----------
    confirm:
        Must be ``True`` to actually discard the ring.  ``False`` (default)
        returns the pre-mutation inventory (``_INTERIM_DISCARD_UNCONFIRMED_STATUS``)
        without mutating anything — the operator's own invocation is the only
        source for this value; the system cannot derive it.
    """

    confirm: bool = False


class InterimDiscardResponse(BaseModel):
    """Response body for ``POST /interim/discard``.

    Attributes
    ----------
    status:
        ``"discarded"`` when the ring was non-empty and was destroyed;
        ``"noop_empty_ring"`` when there was nothing to discard.
    discarded_tiers:
        Interim tier names (``episodic_interim_<stamp>``) dropped from the
        :class:`~paramem.memory.store.MemoryStore`.
    unloaded_adapters:
        PEFT adapter names deleted from the live model (the reaper's return
        value) — empty in the simulate/non-PEFT venue.
    removed_dirs:
        On-disk interim slot directory names removed (``interim_<stamp>``).
    active_keys_destroyed:
        Per-tier count of active keys destroyed, keyed by tier name, as
        measured before the mutation.
    stale_keys_destroyed:
        Per-tier count of stale keys destroyed, keyed by tier name, as
        measured before the mutation.
    resolved_incidents:
        Number of ring-lifecycle incidents (``full_consolidation_overdue``,
        ``interim_cap_reached``, ``interim_overflow_pending``) transitioned
        to ``resolved`` by this call.
    """

    status: str
    discarded_tiers: list[str]
    unloaded_adapters: list[str]
    removed_dirs: list[str]
    active_keys_destroyed: dict[str, int]
    stale_keys_destroyed: dict[str, int]
    resolved_incidents: int


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
        LoRA shape field name: ``"rank"``, ``"alpha"``, or
        ``"target_modules"``. ``dropout`` is never emitted here — it is
        training-time regularization, not tensor shape, so it is excluded
        from the comparison (see ``paramem.server.migration.compute_shape_changes``).
    old_value:
        Value in the on-disk ``meta.json``, or ``None`` when unavailable.
    new_value:
        Value requested by the candidate config.
    consequence:
        Human-readable consequence string (e.g. "restart required", "port in use").
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
        ``None`` when no pre-flight check fires; ``"disk_pressure"`` (over
        the backup store's global cap) or ``"check_error"`` (the check
        itself raised) when a pre-flight check rejects the preview.  Always
        present in the response so callers can check the field
        unconditionally.
    warnings:
        Human-readable rows for adapter tiers skipped during shape-change
        detection: an unreadable/undecryptable tier registry, an unreadable
        adapter manifest, or a tier with one or more on-disk candidate slots
        where none is readable or matches the live registry hash.  Always
        present; empty list when nothing was skipped.
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
    warnings: list[str] = []


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
    restart_eligible:
        ``True`` when an R-PORT carve pre-flighted successfully and the CLI
        may trigger a prompted restart via the ``restart_hint`` command.
        ``False`` for R-PATHS (data-not-migrated warning; operator-driven)
        and for failures.  The server does NOT fire the restart — the CLI
        prompts the operator and, on consent, runs a fixed
        ``systemctl --user restart paramem-server`` via the
        ``paramem.utils.systemctl`` transport seam; ``restart_hint`` is
        display-only text, never the command actually executed.
    cloud_only_reason:
        The reload primitive's own reason when a reload was attempted and
        failed or was refused (one of ``_live_reload_base_model``'s closed
        vocabulary: ``"insufficient_vram"``, ``"reload_failed"``,
        ``"apply_failed"``, ``"config_refused"``).  ``None`` when the apply
        was never attempted (see ``restart_required_reason``) or when it
        succeeded.
    """

    state: str
    trial_adapter_archive_path: str
    restart_required: bool
    restart_hint: str
    pre_migration_backup_retained: bool
    applied_live: bool = False
    restart_required_reason: str | None = None
    restart_eligible: bool = False
    cloud_only_reason: str | None = None


class RollbackResponse(BaseModel):
    """Response body for ``POST /migration/rollback``.

    Attributes
    ----------
    state:
        Always ``"LIVE"`` on success (A config is restored).
    trial_adapter_archive_path:
        Absolute path to the trial adapter archive slot directory (or the
        still-in-place state/trial/adapters/ when rotation failed — 207).
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
    restart_eligible:
        ``True`` when an R-PORT carve pre-flighted successfully and the CLI
        may trigger a prompted restart via the ``restart_hint`` command.
        ``False`` for R-PATHS and for failures.
    cloud_only_reason:
        The reload primitive's own reason when a reload was attempted and
        failed or was refused (one of ``_live_reload_base_model``'s closed
        vocabulary: ``"insufficient_vram"``, ``"reload_failed"``,
        ``"apply_failed"``, ``"config_refused"``).  ``None`` when the apply
        was never attempted (see ``restart_required_reason``) or when it
        succeeded.
    """

    state: str
    trial_adapter_archive_path: str
    rollback_pre_mortem_backup_path: str
    restart_required: bool
    restart_hint: str
    applied_live: bool = False
    restart_required_reason: str | None = None
    restart_eligible: bool = False
    cloud_only_reason: str | None = None


# --- Adapter manifest validation + mount helpers ---
#
# The boot-time validator (_mount_adapters_from_slots) and the post-full-cycle
# revalidator (_revalidate_adapter_manifests) share the same per-tier
# decision logic — extracted into _validate_adapter_slot below so there
# is a single source of truth for "what does this slot's manifest say about
# its health, and should it be mounted?" Main tiers (episodic/semantic/
# procedural) and interim tiers (episodic_interim_*) both route through it —
# interim slots are episodic-shaped, so their fingerprint reference is
# config.adapters.episodic, but each interim's own per-tier registry hash
# (never the main-episodic hash) drives its live-slot match.


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


def _validate_adapter_slot(
    name: str,
    adapter_cfg,
    model,
    kind_dir: Path,
    binding: "TierBinding",
    manifest_status: dict,
) -> "tuple[Path | None, object | None, bool]":
    """Validate one adapter's live slot — main tier or interim tier alike.

    Single source of truth for the per-tier validation decision.  Updates
    ``manifest_status[name]`` with a row for unhealthy outcomes; pops any
    prior row for healthy outcomes (so post-cycle revalidation clears the
    stale boot-time snapshot).

    ``kind_dir`` is the already-resolved tier root — the caller passes it in
    rather than this function re-deriving it from ``name``. This matters for
    interim tiers specifically: re-deriving via
    :func:`~paramem.memory.interim_adapter.adapter_slot_root_for_name` (which
    dispatches interim names to ``interim_dir_for_name``) raises
    ``ValueError`` on a stray ``episodic/interim_<malformed-stamp>/``
    directory, because ``interim_dir_for_name`` demands a well-formed stamp
    while :func:`~paramem.memory.interim_adapter.iter_interim_dirs` yields
    such directories unvalidated (see its docstring, and
    :func:`~paramem.memory.interim_adapter.unload_interim_adapters`'s
    "never re-derived via interim_dir_for_name" comment, which pins the same
    hazard on the reap path). Callers must always pass the exact path
    ``iter_interim_dirs`` yielded (interim case) or
    ``adapter_slot_root_for_name(config.adapter_dir, name)`` (main-tier
    case, where the name is a known-good literal) — never re-derive the
    root from an interim name inside this function.

    ``binding`` is the caller's already-resolved
    :class:`~paramem.adapters.registry_binding.TierBinding` for this exact
    ``kind_dir`` (:func:`~paramem.adapters.registry_binding.verify_tier_binding`).
    It carries the hash-match decision, the resolved slot, and the candidate
    count — this function no longer re-derives any of those.

    Returns ``(slot, manifest, should_mount)``:
      * ``slot``: resolved live slot Path, or ``None`` when no matching slot.
      * ``manifest``: ``binding.manifest`` — already parsed inside
        :func:`~paramem.adapters.registry_binding.verify_tier_binding`, never
        re-read here (a second read could observe a different file than the
        one the verdict was computed from). ``None`` unless ``binding.status``
        is :data:`~paramem.adapters.registry_binding.VERIFIED`.
      * ``should_mount``: True when the boot caller should mount this slot.
        False for "no slot," "registry unverified," "active keys with no
        slot candidate," "key-count mismatch," "payload digest mismatch,"
        "fingerprint mismatch," or a verified slot whose payload kind is
        ``"simulate"`` — a bound simulate slot is healthy (``slot`` and
        ``manifest`` are still returned) but there are no PEFT weights to
        mount.

    Used by:
      * :func:`_mount_adapters_from_slots` (boot path) — for both the main
        tiers and the interim tiers; uses the return triple to decide what
        to mount.
      * :func:`_revalidate_adapter_manifests` (post-full-cycle) — only
        the side-effect on ``manifest_status`` matters; return value is
        discarded.
    """
    from paramem.adapters.registry_binding import (
        KEY_COUNT_MISMATCH,
        KEYS_WITHOUT_SLOT,
        NO_CANDIDATES,
        NO_MATCHING_SLOT,
        PAYLOAD_MISMATCH,
        REGISTRY_ABSENT_WITH_SLOTS,
        REGISTRY_UNREADABLE,
        VERIFIED,
    )
    from paramem.backup.backup import sweep_orphan_pending
    from paramem.server.manifest_status import ROW_STATUS_FOR_VERDICT

    severity = "red" if _is_primary_adapter(name) else "yellow"

    if kind_dir.exists():
        sweep_orphan_pending(kind_dir)

    if binding.status == NO_CANDIDATES:
        manifest_status.pop(name, None)
        logger.info("Adapter %s: no slots found — fresh install", name)
        return None, None, False

    if binding.status == KEYS_WITHOUT_SLOT:
        _record_manifest_row(
            manifest_status,
            name,
            ROW_STATUS_FOR_VERDICT[KEYS_WITHOUT_SLOT],
            "keys_without_slot",
            severity,
        )
        logger.error(
            "Adapter %s: registry holds active keys but no written payload slot "
            "candidate exists (%s) — skipping mount",
            name,
            binding.detail,
        )
        return None, None, False

    if binding.status == NO_MATCHING_SLOT:
        _record_manifest_row(
            manifest_status,
            name,
            ROW_STATUS_FOR_VERDICT[NO_MATCHING_SLOT],
            "no_matching_slot",
            severity,
        )
        logger.warning("Adapter %s: no slot matching registry hash — skipping mount", name)
        return None, None, False

    if binding.status in (REGISTRY_UNREADABLE, REGISTRY_ABSENT_WITH_SLOTS):
        reason = (
            "registry_unreadable"
            if binding.status == REGISTRY_UNREADABLE
            else "registry_absent_with_slots"
        )
        _record_manifest_row(
            manifest_status, name, ROW_STATUS_FOR_VERDICT[binding.status], reason, severity
        )
        logger.error(
            "Adapter %s: registry binding unverified (%s: %s) — skipping mount",
            name,
            reason,
            binding.detail,
        )
        return None, None, False

    if binding.status == KEY_COUNT_MISMATCH:
        _record_manifest_row(
            manifest_status,
            name,
            ROW_STATUS_FOR_VERDICT[KEY_COUNT_MISMATCH],
            "key_count_mismatch",
            severity,
            binding.slot,
        )
        logger.warning(
            "Adapter %s: manifest key_count disagrees with registry active count "
            "(%s) — skipping mount",
            name,
            binding.detail,
        )
        return None, None, False

    if binding.status == PAYLOAD_MISMATCH:
        _record_manifest_row(
            manifest_status,
            name,
            ROW_STATUS_FOR_VERDICT[PAYLOAD_MISMATCH],
            "payload_mismatch",
            severity,
            binding.slot,
        )
        logger.error(
            "Adapter %s: bound slot's payload bytes no longer match the manifest "
            "digest (%s) — skipping mount",
            name,
            binding.detail,
        )
        return None, None, False

    if binding.status != VERIFIED:
        # Every verdict registry_binding.py documents today is handled by
        # name above. A future verdict added there without a matching
        # branch here must not silently fall into the VERIFIED handling
        # below and dereference binding.manifest (None for every
        # non-VERIFIED verdict) — leave it unpublishable loudly instead of
        # crashing boot.
        _record_manifest_row(
            manifest_status,
            name,
            "unrecognized_verdict",
            f"unrecognized_verdict:{binding.status}",
            severity,
            binding.slot,
        )
        logger.error(
            "Adapter %s: unrecognized binding verdict %r — skipping mount",
            name,
            binding.status,
        )
        return None, None, False

    # binding.status == VERIFIED — the manifest is already parsed on the
    # binding; verify_tier_binding only returns VERIFIED after a successful
    # read, so this is never None here.
    slot = binding.slot
    manifest = binding.manifest

    if manifest.payload.kind == "simulate":
        # A simulate payload verifies against the registry exactly like a
        # train one, but there are no PEFT weights to mount — the
        # fingerprint checks below (base_model/tokenizer/lora) apply only to
        # a train payload; a simulate manifest carries all three as None by
        # construction. A bound, verified simulate slot is healthy, not
        # degraded, so no row is recorded — only pop any stale prior row.
        manifest_status.pop(name, None)
        return slot, manifest, False

    mismatch_field = _check_manifest_fingerprints(manifest, model, adapter_cfg)
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
        # Red is only correct for the primary tier: a red row renders as a
        # failed-level "... — PA routing DISABLED" attention item
        # (paramem/server/attention.py:550-560,
        # _collect_adapter_fingerprint_items), and PA routing is only
        # disabled by a primary-tier problem. A non-synthesized manifest on
        # a non-primary tier (semantic/procedural/interim) that still has
        # UNKNOWN fields is unexpected but must stay yellow/info, matching
        # every other severity decision in this function
        # (severity = "red" if _is_primary_adapter(name) else "yellow").
        unknown_severity = (
            "red" if _is_primary_adapter(name) and not manifest.synthesized else "yellow"
        )
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


def _revalidate_adapter_manifests(state: dict) -> None:
    """Re-run :func:`_validate_adapter_slot` for every tier — main AND
    interim — and refresh ``state['adapter_manifest_status']``.  The single
    row-freshness owner for the whole ``adapter_manifest_status`` dict,
    called from the two event-kind fold finalizers (``_finalize_full`` and
    ``_finalize_interim``) so rows refresh at every fold, train or
    simulate venue alike (``_finalize_interim`` is the one finalizer for
    every interim-shaped terminal, first-run or resumed, either venue — see
    its own docstring).  A simulate-mode tier's own bound, VERIFIED slot
    mints no row — :func:`_validate_adapter_slot`'s simulate branch pops any
    stale row and reports ``should_mount=False`` without recording one (a
    healthy graph payload has no PEFT weights to mount but is not degraded).
    Every OTHER binding verdict (``KEYS_WITHOUT_SLOT``, ``NO_MATCHING_SLOT``,
    ``KEY_COUNT_MISMATCH``, ``PAYLOAD_MISMATCH``, ``REGISTRY_UNREADABLE``,
    ``REGISTRY_ABSENT_WITH_SLOTS``) mints a row exactly like a train-mode
    tier's would — none of those branches inspect ``payload.kind`` at all —
    so a simulate-mode cycle CAN mint a row (e.g. a corrupted ``graph.json``
    after write). Calling this on a simulate terminal is therefore not
    always a no-op; any row that predates a switch to simulate mode still
    gets refreshed here exactly like a train terminal would, and a fresh
    simulate-venue failure gets its own row the same way a train one does.

    Boot's :func:`_mount_adapters_from_slots` snapshots adapter health from
    the on-disk state at startup.  After a fold re-saves a tier's slot with
    a fresh registry hash, that boot-time snapshot is stale — operators see
    ``FINGERPRINT MISMATCH … PA routing DISABLED`` on /status / pstatus
    even though the tier is healthy.  Calling this from both training
    finalizers clears those stale rows on every training fold, main or
    interim.

    NOT pure validation: :func:`_validate_adapter_slot` calls
    ``sweep_orphan_pending(kind_dir)`` (backup/atomic.py), which
    unconditionally deletes everything under ``<kind_dir>/.pending/`` — the
    same staging directory ``atomic_save_adapter`` writes into before its
    rename-into-place. That is safe here only positionally: both callers
    dispatch this function AFTER the fold's own saves have completed, and
    ``_state["consolidating"]`` is still ``True`` for the whole window,
    blocking every other consolidation door from starting a competing save
    into the same ``.pending/`` dir. Do not move this call earlier in
    either finalizer, and do not call it from anywhere that runs
    concurrently with an in-flight save, on the strength of a "pure
    validation" reading of this function — ``model`` is read but never
    mutated, and nothing is ever mounted, but the pending-dir sweep is a
    real, unconditional delete. Healthy adapters have their row removed;
    unhealthy ones get a fresh row stamped with the current ``checked_at``.

    Main tiers first — every tier in ``config.tier_config_map()`` — then
    every interim dir :func:`iter_interim_dirs` currently yields (interims
    stay episodic-shaped and validated regardless of the main episodic
    tier's enabled state, subject to the boot-time refusal of a populated
    ring left behind by a disabled episodic tier — see
    :func:`_load_model_into_state`; ``kind_dir`` is the
    exact path ``iter_interim_dirs`` yielded, never re-derived — see
    :func:`_validate_adapter_slot`'s docstring for why). Finally, ONE
    post-loop prune drops every ``manifest_status`` row keyed by a name that
    is neither in ``config.tier_config_map()`` nor a live interim directory
    name — this subsumes both a tier disabled since its row was written and
    an interim slot folded away or discarded since (a full cycle can retire
    an interim slot without that interim ever being revalidated again, so
    without this prune a stale row would linger and permanently suppress
    the ``local_recall_inactive`` attention item — any row with a
    problematic status defers to the more specific fingerprint item instead,
    see the guard in
    :func:`~paramem.server.attention._collect_local_recall_inactive_items`).
    """
    config = state.get("config")
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    if config is None or model is None or tokenizer is None:
        return

    from paramem.adapters.registry_binding import verify_tier_binding
    from paramem.memory.interim_adapter import (
        adapter_slot_root_for_name,
        iter_interim_dirs,
    )

    manifest_status = state.setdefault("adapter_manifest_status", {})

    _tier_configs = config.tier_config_map()
    for name, adapter_cfg in _tier_configs.items():
        _kind_dir = adapter_slot_root_for_name(config.adapter_dir, name)
        _validate_adapter_slot(
            name,
            adapter_cfg,
            model,
            _kind_dir,
            verify_tier_binding(name, _kind_dir),
            manifest_status,
        )

    live_interims: list = []
    if "episodic" in _tier_configs:
        live_interims = list(iter_interim_dirs(config.adapter_dir))
        for _interim_name, _interim_path in live_interims:
            _validate_adapter_slot(
                _interim_name,
                _tier_configs["episodic"],
                model,
                _interim_path,
                verify_tier_binding(_interim_name, _interim_path),
                manifest_status,
            )

    live_names = set(_tier_configs) | {n for n, _ in live_interims}
    for stale_name in [n for n in manifest_status if n not in live_names]:
        manifest_status.pop(stale_name, None)


def _consolidation_terminal(body: "Callable[[], None] | None" = None) -> None:
    """The single exit of every consolidation-envelope run.

    Runs *body* — the run's own bookkeeping (manifest revalidation, incident
    sweeps, router reload, retirement, the run-status row, the
    ``_state["calibration_run"]`` outcome) — on the asyncio event loop, then
    clears ``_state["consolidating"]`` in a ``finally`` so a raising body
    still releases the arbitrator while its exception still propagates to
    the loop's handler.  ``None`` is a run whose terminal has no
    bookkeeping.

    The post-run state mutations (``_state["last_consolidation*"]``, the
    router reload, and clearing ``_state["consolidating"]``) must be visible
    to /chat handlers atomically with the consolidating-flag clear. They run
    on the BG-trainer worker thread or a calibration executor thread, so
    when the event loop is live we hand the closure to it via
    ``call_soon_threadsafe``; otherwise (no loop / tests) we invoke it
    inline on the current thread.

    Scope: runs dispatched through :func:`_dispatch_to_executor`.  The three
    doors that borrow the same flag as an in-handler mutex — ``POST
    /interim/discard``, ``POST /speaker/forget``, ``POST /debug/erase-keys``
    — keep their own request-local ``try/finally``: their no-await-tail
    invariant requires the clear to land with no yield point, which posting
    to the loop would break.
    """

    def _run() -> None:
        try:
            if body is not None:
                body()
        finally:
            _state["consolidating"] = False

    aio_loop = _state.get("event_loop")
    if aio_loop is not None and aio_loop.is_running():
        aio_loop.call_soon_threadsafe(_run)
    else:
        _run()


def _sweep_keyless_tier_artifacts(config, state: dict) -> list[str]:
    """Reap every tier whose registry↔slot binding legitimately reads as empty.

    Pre-mount housekeeping, in order:

    1. :func:`~paramem.memory.persistence.resume_pending_reaps` finishes any
       tier-artifact deletion a prior :func:`reap_tier_artifacts` call left
       stranded under ``.pending-delete/`` (crash between the rename-condemn
       and the actual delete). A stranded tombstone is already out of the
       live namespace, so nothing below can see or reap it; only the
       tombstone-specific resume can finish it.
    2. :func:`~paramem.backup.integrity.cleanup_partial_slots` deletes any
       ``<adapter_dir>/<tier>/<slot>/`` scratch directory missing one of the
       three canonical slot files (interrupted training write). This runs
       BEFORE any tier's binding is verified below, and before mount
       validation / store publish / the integrity report run later in the
       same boot (all reached only after this function returns) — a torn
       scratch dir left in place could otherwise be counted as a candidate
       slot (:func:`~paramem.adapters.manifest.count_slot_candidates` only
       requires a readable ``meta.json``, not a complete slot) and produce a
       verdict that disagrees with what a later, post-cleanup pass would
       compute for the same tier. Any removal is recorded on
       ``state["integrity_cleanup"]`` for the attention populator — *state*
       is required (not optional) precisely so this bookkeeping is never
       silently skipped; direct-call tests pass an explicit ``{}`` when the
       recorded value is not under test.
    3. The keyless-tier scan below: every main + interim tier root
       (:func:`~paramem.memory.interim_adapter.iter_tier_roots`) is
       evaluated via :func:`~paramem.adapters.registry_binding.verify_tier_binding`
       — the single shape/read oracle for this stage (no separate
       ``KeyRegistry.load``/``load_simhashes`` call here; a raise inside
       ``verify_tier_binding`` itself already resolves to
       :data:`~paramem.adapters.registry_binding.REGISTRY_UNREADABLE`).

    Runs pre-mount, at the top of :func:`_mount_adapters_from_slots` — before
    any slot is resolved for any tier, before ``find_live_slot`` is called,
    and before :func:`~paramem.adapters.registry_binding.verify_tier_binding`
    reads any tier's registry a second time for mounting. Nothing this sweep
    removes is ever mounted, and it never unmounts a live adapter — at this
    point in boot nothing has been mounted yet. The memory store is hydrated
    later still (the lifespan calls ``_build_runtime_components`` only
    after ``_load_model_into_state`` returns), so no RAM-resident tier can
    ever outlive the files this sweep removed. The same call also runs on
    every live reload — :func:`_live_reload_base_model` reaches this
    function through its own call to :func:`_load_model_into_state` — but
    never on a cloud-only boot, because ``_load_model_into_state`` is only
    invoked when a local model is about to be loaded; a cloud-only server
    serves nothing from any tier and never reaches this code. (A cloud-only
    boot still runs ``_build_store_contents``'s own ``verify_adapter_tree``
    call for the memory store — unconditionally, regardless of mode — so
    that pass sees whatever partial-slot state is already on disk; the next
    boot or reload that acquires a local model runs this sweep, including
    :func:`~paramem.backup.integrity.cleanup_partial_slots`, before any
    binding is computed and self-heals it then.)

    Operator erase doors (``POST /speaker/forget``, ``POST /debug/erase-keys``) no longer
    empty a tier's registry — they stale-mark, withholding an ACTIVE key as
    a marker that keeps its id in ``list_known()`` (see
    :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`), so
    ``list_known()`` never drops as a side effect of an operator erase. The
    one legitimate emptier of a tier's ``list_known()`` is that tier's own
    rebuild, which seeds the fold's working copy from active keys alone and
    so publishes a registry carrying no marker (item 4 below is exactly that
    shape: a matched, zero-``key_count`` manifest). A "registry says empty,
    slot binding does not independently corroborate it" shape at boot can
    therefore only be a torn commit or genuine corruption — never a
    legitimately-interrupted erase, and never an unmatched rebuild, which
    would show as item 4's clean match instead — and is always preserved
    (never reaped) below; the operator's recovery door is
    ``POST /backup/restore``.

    Per tier root, after ``binding = verify_tier_binding(name, root)``:

    1. ``binding.registry is None`` (:data:`~paramem.adapters.registry_binding.REGISTRY_UNREADABLE`
       — corrupt file, failed decrypt, or any other read failure) — preserved
       and logged as an ERROR, regardless of the marker. An unreadable
       registry is never inferred to hold zero keys; only a registry that
       actually answers and says so is swept.
    2. ``binding.registry.list_known()`` (active ∪ stale) non-empty —
       preserved. When the tier carries at least one ACTIVE key and no slot
       candidate at all in either venue (``binding.status ==
       KEYS_WITHOUT_SLOT``, i.e. ``binding.candidate_count == 0`` — no
       payload at all) this is logged as an ERROR naming the known-key
       count — the same shape a crash-interrupted erase or a torn training
       write can both produce, and deleting facts that were never folded
       anywhere else would be a silent data loss. Every other non-empty
       shape (a bound slot in either venue, a keyed main tier still
       awaiting a matching slot, or a registry whose only known keys are
       STALE with zero candidates — still ``NO_CANDIDATES`` under the
       active-key-gated split) is preserved SILENTLY — a tier that still
       knows keys is not this sweep's to report; the mount stage mints its
       own row and log line for whatever binding status a keyed tier
       resolves to, and this keeps that to one reporter per condition.
    3. ``list_known()`` empty and the registry file does not exist
       (``binding.registry_present is False``):
       :data:`~paramem.adapters.registry_binding.NO_CANDIDATES` — skipped
       silently (fresh install, nothing on disk to reap);
       :data:`~paramem.adapters.registry_binding.VERIFIED` (a
       ``""``-stamped slot whose manifest agrees the tier is empty, stamping
       ``key_count`` as ``0`` or leaving it
       :data:`~paramem.adapters.manifest.UNKNOWN`) — skipped silently (the
       experiment-install shape);
       :data:`~paramem.adapters.registry_binding.REGISTRY_ABSENT_WITH_SLOTS`
       — preserved and logged as an ERROR, UNCONDITIONALLY: this is the torn
       weight-before-registry commit shape shared by both per-tier commit
       primitives (:func:`commit_tier_slot`'s migration/trial-tree callers
       and the fold path's :func:`write_tier_slot` + :func:`publish_tier_registry`
       — a crash after the weight write but before the registry flush; see
       either function's crash-semantics note), and a reap here would
       discard a slot whose registry commit never landed; a ``""``-stamped
       slot whose manifest disagrees on
       ``key_count`` (:data:`~paramem.adapters.registry_binding.KEY_COUNT_MISMATCH`)
       or the rare manifest-read-race
       :data:`~paramem.adapters.registry_binding.NO_MATCHING_SLOT` falls
       through to the unconditional-preserve fallback below.
    4. ``list_known()`` empty and the registry file exists
       (``binding.registry_present is True``):
       :data:`~paramem.adapters.registry_binding.NO_CANDIDATES` — reaped
       unconditionally (a registry file and nothing else — no manifest to
       read at all, so there is no visible cross-artifact claim to weigh
       against the registry's own empty read; the ordinary self-heal this
       sweep has always performed);
       :data:`~paramem.adapters.registry_binding.VERIFIED` with an
       ``int`` ``manifest.key_count == 0`` — reaped unconditionally
       (registry and matched manifest independently agree the tier is
       empty). Everything else —
       :data:`~paramem.adapters.registry_binding.NO_MATCHING_SLOT`
       (deliberately NOT reaped unconditionally here, unlike
       :data:`NO_CANDIDATES`: ``find_live_slot`` returns the same verdict
       whether every candidate's hash genuinely mismatched or a candidate's
       ``meta.json`` was merely unreadable — this status cannot tell a
       stale slot from a transiently corrupt one, so it never gets the
       benefit of the doubt),
       :data:`~paramem.adapters.registry_binding.KEY_COUNT_MISMATCH`
       (a matching-hash slot whose ``key_count`` disagrees), or a
       ``VERIFIED`` match whose ``key_count`` is
       :data:`~paramem.adapters.manifest.UNKNOWN` — falls through to the
       unconditional-preserve fallback below.
    5. Fallback (every ``list_known()``-empty shape not resolved by 3 or 4
       above): always preserved and logged as an ERROR, naming
       ``POST /backup/restore`` as the recovery door (never
       ``/reconsolidate``, which cannot rebuild a tier whose registry
       binding is itself unverified). No caller can legitimately produce
       this shape — the operator erase doors stale-mark rather than empty a
       tier — so there is nothing left to authorise a reap against; a torn
       commit or genuine corruption is the only remaining explanation.

    Args:
        config: Loaded ``ServerConfig``; only ``adapter_dir`` is read.
        state: The global ``_state`` dict (or an explicit ``{}`` from a
            direct-call test), for recording
            :func:`~paramem.backup.integrity.cleanup_partial_slots` removals
            on ``state["integrity_cleanup"]``. Required, not optional —
            that bookkeeping must never be silently skipped.

    Returns:
        Sorted tier names whose artifacts were actually removed by the
        keyless-tier scan (step 3 above) — NOT tiers whose scratch was
        removed by :func:`~paramem.backup.integrity.cleanup_partial_slots`.
    """
    from paramem.adapters.registry_binding import (
        KEYS_WITHOUT_SLOT,
        NO_CANDIDATES,
        REGISTRY_ABSENT_WITH_SLOTS,
        VERIFIED,
        verify_tier_binding,
    )
    from paramem.backup.integrity import cleanup_partial_slots
    from paramem.memory.interim_adapter import iter_tier_roots
    from paramem.memory.persistence import (
        reap_tier_artifacts,
        resume_pending_reaps,
    )

    resume_pending_reaps(config.adapter_dir)

    _partial_removed = cleanup_partial_slots(config.adapter_dir)
    if _partial_removed:
        logger.warning(
            "Boot sweep: cleanup_partial_slots removed %d partial slot(s) "
            "before any tier's registry binding is verified",
            len(_partial_removed),
        )
        state["integrity_cleanup"] = _partial_removed

    roots: list[tuple[str, Path]] = list(iter_tier_roots(config.adapter_dir))

    reaped: list[str] = []
    for name, root in roots:
        binding = verify_tier_binding(name, root)

        if binding.registry is None:
            logger.error(
                "Boot sweep: tier %s registry binding is unreadable (%s) — preserving",
                name,
                binding.detail,
            )
            continue

        known = binding.registry.list_known()
        if known:
            if binding.status == KEYS_WITHOUT_SLOT:
                logger.error(
                    "Boot sweep: tier %s has a registry with %d known key(s) but "
                    "no slot candidate at all (candidate_count=%d, no payload in "
                    "either venue) — preserving; recover via consolidation fold "
                    "or registry restore",
                    name,
                    len(known),
                    binding.candidate_count,
                )
            # Every other status (VERIFIED, KEY_COUNT_MISMATCH,
            # NO_MATCHING_SLOT, REGISTRY_ABSENT_WITH_SLOTS, PAYLOAD_MISMATCH,
            # or NO_CANDIDATES with only STALE known keys) means either at
            # least one slot candidate exists for this tier in either venue,
            # or no active key needs one: the tier's content, if any, is not
            # this sweep's to report — the mount stage mints its own row and
            # log line for whatever binding status a keyed tier resolves to.
            # One reporter per condition.
            continue

        # known is empty from here on. One reap primitive for every arm below
        # that decides to reap — a shared closure so a future arm cannot add
        # a reap that forgets `reaped.append`.
        def _reap(reason: str) -> None:
            removed = reap_tier_artifacts(root)
            if removed:
                reaped.append(name)
                logger.warning(
                    "Boot sweep: tier %s reaped — %s — removed %d stale "
                    "artifact path(s) (self-heals a torn commit or genuine "
                    "corruption — never an operator erase, which stale-marks "
                    "rather than empties a registry)",
                    name,
                    reason,
                    len(removed),
                )

        # The decision table's two halves, structurally: registry-absent
        # shapes on one side, registry-present shapes on the other. Neither
        # branch falls all the way through on its own — every arm either
        # `continue`s or drops out of the if/else to the shared
        # unconditional-preserve fallback below.
        if not binding.registry_present:
            if binding.status == NO_CANDIDATES:
                continue  # fresh install — nothing on disk to reap
            if binding.status == VERIFIED:
                # A ""-stamped slot whose manifest independently agrees the
                # tier is empty (key_count 0 or UNKNOWN) — the
                # experiment-install shape. Nothing to reap, nothing to log.
                continue
            if binding.status == REGISTRY_ABSENT_WITH_SLOTS:
                logger.error(
                    "Boot sweep: tier %s has %d candidate slot(s) but no "
                    "indexed_key_registry.json at all (%s) — preserving; "
                    "this is the torn weight-before-registry commit shape "
                    "(registry flush interrupted after the weight write, "
                    "from either per-tier commit primitive) — recover "
                    "via POST /backup/restore then restart",
                    name,
                    binding.candidate_count,
                    binding.detail,
                )
                continue
            # KEY_COUNT_MISMATCH (a ""-stamped slot whose manifest disagrees
            # on key_count), PAYLOAD_MISMATCH (a ""-stamped slot whose
            # payload bytes disagree with its own manifest digest), or the
            # rare manifest-read-race NO_MATCHING_SLOT falls through to the
            # unconditional-preserve fallback below.
        else:
            if binding.status == NO_CANDIDATES:
                # A registry file and nothing else — no manifest to read at
                # all, so there is no visible cross-artifact claim to weigh
                # against the registry's own empty read; the ordinary
                # self-heal this sweep has always performed.
                _reap("registry lists zero known keys with no written payload slot candidate")
                continue
            if (
                binding.status == VERIFIED
                and isinstance(binding.manifest.key_count, int)
                and binding.manifest.key_count == 0
            ):
                # Registry and the matched slot's manifest independently
                # agree the tier is empty — reap unconditionally.
                _reap("registry and matched slot manifest both read zero known keys")
                continue
            # NO_MATCHING_SLOT (deliberately NOT reaped unconditionally,
            # unlike NO_CANDIDATES: find_live_slot returns the same verdict
            # whether every candidate's hash genuinely mismatched or a
            # candidate's meta.json was merely unreadable — this status
            # cannot tell a stale slot from a transiently corrupt one, so it
            # never gets the benefit of the doubt), KEY_COUNT_MISMATCH (a
            # matching-hash slot whose key_count disagrees), PAYLOAD_MISMATCH
            # (a matching-hash slot whose payload bytes disagree with its own
            # manifest digest), or a VERIFIED match whose key_count is
            # UNKNOWN falls through to the unconditional-preserve fallback
            # below.

        # Unconditional-preserve fallback: the registry reads zero known
        # keys, but the binding does not independently, unambiguously
        # corroborate that. Every other empty-known shape was already
        # resolved above. The operator erase doors stale-mark rather than
        # empty a tier, so there is no legitimate producer of this shape
        # left to authorise a reap against — a torn commit or genuine
        # corruption is the only remaining explanation, and both are
        # preserved for the operator to recover.
        logger.error(
            "Boot sweep: tier %s registry reads zero known keys but its "
            "slot binding (%s: %s) does not independently corroborate "
            "that — preserving; restore this tier from a snapshot bundle "
            "via POST /backup/restore and restart; see GET /integrity",
            name,
            binding.status,
            binding.detail,
        )

    return sorted(reaped)


def _record_tier_weight_state(state: dict, model, config) -> None:
    """Snapshot per-main-tier trained-weight status onto ``state``, once.

    ``has_prior_trained_weights`` walks ``model.named_parameters()`` — cheap
    once per adapter mutation, expensive if called on every ``/status`` poll
    (the fold cadence is roughly once per second). Weight state only
    actually changes where an adapter is mounted or promoted, so this is
    called exactly at those boundaries — the end of
    :func:`_mount_adapters_from_slots` (covers both the boot path via
    :func:`_load_model_into_state` and the restore path via
    :func:`_remount_adapters_from_disk`, which both call it) and the
    full-cycle finalizer (:func:`_finalize_full`, the go-live promote path,
    shared by every full/reconcile-shaped fold regardless of which driver
    dispatched it) — never recomputed by a reader. ``/status`` reads the
    recorded map instead of calling ``has_prior_trained_weights`` itself.

    Args:
        state: The global ``_state`` dict (mutated in place).
        model: The live model, or ``None`` (cloud-only) — an empty map is
            recorded in that case.
        config: The live ``ServerConfig`` — ``tier_config_map()`` names the
            tiers to snapshot.
    """
    if model is None:
        state["tier_weight_state"] = {}
        return
    state["tier_weight_state"] = {
        name: has_prior_trained_weights(model, name) for name in config.tier_config_map()
    }


def _mount_adapters_from_slots(model, tokenizer, config, state: dict) -> None:
    """Load enabled adapters from slot-dir layout with manifest verification, in place.

    Pre-mount housekeeping runs first, before any slot is resolved for any
    tier (:func:`_sweep_keyless_tier_artifacts`, which itself runs
    ``resume_pending_reaps`` then
    :func:`~paramem.backup.integrity.cleanup_partial_slots` then the
    keyless-tier scan proper — see that function's docstring for the full
    ordering rationale and why it must precede every consumer of
    :func:`~paramem.adapters.registry_binding.verify_tier_binding` /
    :func:`~paramem.adapters.registry_binding.verify_adapter_tree` reached
    later in the same boot: this mount loop, the memory-store publish
    (``_build_store_contents``, called from ``_preload_memory_store`` further
    down the same boot/reload event), and the integrity report all then see
    the identical, already-cleaned tree instead of three independently timed
    snapshots that could disagree). Then every adapter kind — every tier in
    ``config.tier_config_map()`` AND every interim tier on disk
    (``episodic_interim_*``, gated on ``"episodic" in config.tier_config_map()``
    — the boot-time refusal already blocks boot when episodic is disabled
    with a populated ring, so this is a second, defensive gate rather than
    the primary one)
    — is validated through the single :func:`_validate_adapter_slot`
    decision tree:

    1. Sweep orphan ``.pending`` dirs (inside the validator, scoped to that
       tier's own slot root).
    2. Resolve the tier's :class:`~paramem.adapters.registry_binding.TierBinding`
       (per-tier: main tiers verify their own registry; interim tiers verify
       their own, never the main-episodic one) via
       :func:`~paramem.adapters.registry_binding.verify_tier_binding`.
    3. The binding carries the matched slot, when one was found.
    4. Read the manifest; compare base model / tokenizer / LoRA fingerprints
       (interim tiers compare against ``config.tier_config_map()["episodic"]``
       — interim slots are episodic-shaped).
    5. Mount matching slots; record mismatch / missing / unverified rows in
       ``state["adapter_manifest_status"]``.

    Finally, :func:`_record_tier_weight_state` snapshots each main tier's
    trained-weight status onto ``state["tier_weight_state"]`` — the one
    write ``/status`` reads instead of measuring on every poll.

    Args:
        model: The live ``PeftModel`` to mount adapters onto. Every tier is
            already resident (created cold by :func:`load_base_model` /
            :func:`~paramem.models.loader.ensure_resident_tiers`) — this
            function only ever mounts trained weights onto an existing
            adapter name, never creates one. Mutated in place; nothing is
            returned.
        tokenizer: Loaded tokenizer (for fingerprint comparison).
        config: Loaded ``ServerConfig``.
        state: The global ``_state`` dict (mutated in-place for manifest
            status, ``state["tier_weight_state"]`` and, when
            :func:`_sweep_keyless_tier_artifacts`'s
            ``cleanup_partial_slots`` pass removes anything,
            ``state["integrity_cleanup"]``).
    """
    from paramem.adapters.registry_binding import verify_tier_binding
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.models.loader import mount_adapter

    manifest_status: dict = state.setdefault("adapter_manifest_status", {})
    # Per-tier paths live at <adapter_dir>/<tier>/indexed_key_registry.json; each
    # tier's slot manifest is stamped with that tier's own registry hash, so
    # slot matching is per-tier (see verify_tier_binding).

    # Self-heal any tier whose on-disk registry has already dropped to zero
    # known keys but whose slot directory/manifest still lingers (crash
    # window between a hard key erase and its reap, or a torn training
    # write). Must run BEFORE any slot below is resolved or mounted — see
    # _sweep_keyless_tier_artifacts's docstring.
    _swept_tiers = _sweep_keyless_tier_artifacts(config, state)
    if _swept_tiers:
        logger.info("Boot sweep: reaped keyless tier(s): %s", ", ".join(_swept_tiers))

    def _load_one(name: str, slot: Path):
        """Mount a single adapter from *slot* onto *model*, in place.

        Does not overwrite an existing manifest status row — the validator may
        have already recorded a manifest_missing or migrated_unverified row.
        """
        try:
            mount_adapter(model, slot, name)
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
    # Per-tier validation is delegated to _validate_adapter_slot so the
    # boot path and post-full-cycle revalidation share one decision tree.
    _tier_configs = config.tier_config_map()
    for name, adapter_cfg in _tier_configs.items():
        _kind_dir = adapter_slot_root_for_name(config.adapter_dir, name)
        slot, _manifest, should_mount = _validate_adapter_slot(
            name,
            adapter_cfg,
            model,
            _kind_dir,
            verify_tier_binding(name, _kind_dir),
            manifest_status,
        )
        if should_mount and slot is not None:
            _load_one(name, slot)

    # ---- Interim adapters ----
    # Same _validate_adapter_slot decision tree as the main tiers, using
    # tier_config_map()["episodic"] as the fingerprint reference (interim
    # slots are episodic-shaped) and each interim's own per-tier registry
    # hash (never the main-episodic hash — comparing to the main hash always
    # misses when a full cycle hasn't run yet). Gated on episodic existing —
    # a populated ring under a disabled episodic tier is refused at boot
    # before this function is ever reached; this is belt-and-suspenders
    # for any other caller of this function (e.g. a reload).
    # ``_interim_path`` (the exact path iter_interim_dirs yielded) is passed
    # as kind_dir directly — never re-derived from ``_interim_name`` via
    # adapter_slot_root_for_name, which raises on a stray dir whose stamp is
    # malformed (see _validate_adapter_slot's docstring).
    if "episodic" in _tier_configs:
        from paramem.memory.interim_adapter import iter_interim_dirs

        for _interim_name, _interim_path in iter_interim_dirs(config.adapter_dir):
            slot, _manifest, should_mount = _validate_adapter_slot(
                _interim_name,
                _tier_configs["episodic"],
                model,
                _interim_path,
                verify_tier_binding(_interim_name, _interim_path),
                manifest_status,
            )
            if should_mount and slot is not None:
                _load_one(_interim_name, slot)

    logger.info("Adapters loaded: %s", list(model.peft_config.keys()))
    _record_tier_weight_state(state, model, config)


def _check_manifest_fingerprints(manifest, model, adapter_cfg) -> "str | None":
    """Compare manifest fingerprints against live runtime state.

    TRAIN-payload manifests only — the sole caller,
    :func:`_validate_adapter_slot`, short-circuits on
    ``manifest.payload.kind == "simulate"`` (its own docstring's
    "should_mount" note) BEFORE ever calling this function, so
    ``manifest.base_model``/``manifest.tokenizer``/``manifest.lora`` are
    guaranteed non-``None`` here by the schema invariant
    (:class:`~paramem.adapters.manifest.AdapterManifest`'s own
    ``__post_init__``: ``payload.kind == "train"`` requires all three
    present). A ``simulate``-payload manifest never reaches this function at
    all — it does not need a defensive re-check here.

    Skips UNKNOWN values (cannot verify) on the ``base_model`` fields.
    Returns the name of the first mismatching field, or ``None`` when all
    checked fields match.

    LoRA shape: one loop over ``rank``/``alpha``/``target_modules`` pairs
    each field's stamped value against its live ``adapter_cfg`` value.
    ``manifest.synthesized`` decides only whether a falsy stamped value
    (``0``, ``0.0``, or ``()``) is skipped rather than compared:

    * ``synthesized=True`` — a falsy value is skipped (the migration script,
      ``scripts/migrate/outputs_to_slot_dirs.py``, cannot recover LoRA
      hyperparameters for some legacy layouts and legitimately leaves them
      at zero; that manifest also carries ``key_count=UNKNOWN`` and is
      routed to ``migrated_unverified`` by :func:`_first_unknown_field`,
      never silently mounted). A truthy value is compared for equality.
    * ``synthesized=False`` — nothing is skipped: a falsy OR
      non-matching stamped value is a mismatch. A zero/empty field here
      means :func:`~paramem.adapters.manifest.build_manifest_for` found no
      ``model.peft_config`` entry for this adapter at build time — a real
      fingerprint mismatch, not an unrecoverable unknown — and must not
      mount silently (``base_model``/``tokenizer``/``registry_sha256`` are
      typically all known in this shape, so :func:`_first_unknown_field`
      alone would miss it). The strict equality check also means a non-int
      stamped value reaching this function on a non-synthesized manifest
      (a degraded/malformed stamp — schema declares ``lora.rank: int``, so
      this should never happen in practice) always compares unequal to the
      live int and is reported as a mismatch rather than silently passed
      through: the previous ``isinstance(..., int)`` guard, which skipped
      the comparison entirely for a non-int value, is deliberately removed.

    Args:
        manifest: :class:`~paramem.adapters.manifest.AdapterManifest` to check.
        model: Live base model (or PeftModel) with ``config`` attribute.
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

    # LoRA shape — see docstring for the synthesized-gated skip semantics.
    live_targets = tuple(sorted(adapter_cfg.target_modules or []))
    lora_fields = (
        ("lora.rank", manifest.lora.rank, adapter_cfg.rank),
        ("lora.alpha", manifest.lora.alpha, adapter_cfg.alpha),
        ("lora.target_modules", manifest.lora.target_modules, live_targets),
    )
    for field_name, stamped, live in lora_fields:
        if manifest.synthesized and not stamped:
            continue
        if not stamped or stamped != live:
            return field_name

    return None


def _first_unknown_field(manifest) -> "str | None":
    """Return the name of the first UNKNOWN-valued field, or None.

    TRAIN-payload manifests only — the sole caller,
    :func:`_validate_adapter_slot`, short-circuits on
    ``manifest.payload.kind == "simulate"`` BEFORE ever calling this
    function (same short-circuit :func:`_check_manifest_fingerprints`
    documents), so ``manifest.base_model``/``manifest.tokenizer`` are
    guaranteed non-``None`` here by the schema invariant and their checks
    below run unconditionally. Only checks fields that are material for
    adapter verification; ``registry_sha256`` and ``key_count`` are checked
    regardless of payload kind (both fields exist on every manifest).
    """
    from paramem.adapters.manifest import UNKNOWN

    checks: list[tuple[str, "str | int"]] = [
        ("registry_sha256", manifest.registry_sha256),
        ("base_model.repo", manifest.base_model.repo),
        ("base_model.sha", manifest.base_model.sha),
        ("base_model.hash", manifest.base_model.hash),
        ("tokenizer.name_or_path", manifest.tokenizer.name_or_path),
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
    _tier_configs = config.tier_config_map()
    main_adapter_configs = list(_tier_configs.values())
    # Interims are always episodic-shaped, regardless of whether episodic is
    # itself an enabled main tier. ``None`` when episodic is absent — config
    # load already forbids max_interim_count > 0 without episodic, so there
    # is no interim shape to derive and assess_topology treats the interim
    # contribution as 0 in that case (never a fallback shape borrowed from
    # another tier).
    _interim_config = _tier_configs.get("episodic")
    assessment = assess_topology(
        _interim_config,
        main_adapter_configs=main_adapter_configs,
        max_interim_count=config.consolidation.max_interim_count,
        interim_overflow_slack=config.consolidation.interim_overflow_slack,
        base_bytes=base_pred,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lora_dtype_bytes=lora_dtype_bytes,
        peft_overhead_bytes=peft_overhead_bytes,
        baseline_vram_gib=config.vram.baseline_vram_gib,
        model_id=config.model_config.model_id,
        quant_label=config.model_config.quantization,
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

    Store presence is the ONLY auth-enablement signal — the shared
    ``PARAMEM_API_TOKEN`` validation branch is retired; every credential now
    lives exclusively in ``UserTokenStore``.  The store is built when
    EITHER of two things is true:

    * ``config.mobile_pwa.enabled`` — the operator has opted in explicitly, or
    * the store's on-disk file (``config.paths.data / "user_tokens.json"``)
      already exists — a prior ``paramem mint-user-token`` run wrote it, so
      the server must pick up those credentials even if
      ``mobile_pwa.enabled`` is ``False`` in this config.

    A fresh install with neither condition true stays auth-OFF (store is
    ``None``) until the operator's first mint; minting creates the file, and
    ONLY the NEXT boot turns the server ON — the live mtime-reload inside
    :class:`~paramem.server.user_tokens.UserTokenStore` (see its
    ``_maybe_reload``) refreshes an ALREADY-wired store's token set from
    disk, but cannot conjure a store into existence: with no store object
    assigned in ``_state["user_token_store"]`` there is nothing for
    ``_maybe_reload`` to be called on, so a live mint against a from-scratch
    (auth-OFF) deployment does not flip it ON without a restart. This still
    keeps a from-scratch deployment usable immediately, without a
    chicken-and-egg "enable auth to mint a token, mint a token to enable
    auth" step — the cost is a restart after the first mint, not an
    unreachable state.

    The decision logic is extracted here so it can be unit-tested without
    starting the full app lifespan.

    Parameters
    ----------
    config:
        A :class:`~paramem.server.config.ServerConfig` instance.

    Returns
    -------
    UserTokenStore | None
        A wired store (potentially empty) when either condition above holds,
        else ``None``.
    """
    store_path = config.paths.data / "user_tokens.json"
    if not (config.mobile_pwa.enabled or store_path.exists()):
        return None
    from paramem.server.user_tokens import UserTokenStore as _UserTokenStore

    return _UserTokenStore(store_path)


# ---------------------------------------------------------------------------
# CUDA fail-fast helpers — boot-time sticky-context detection and recovery
# ---------------------------------------------------------------------------


def _degrade_to_cloud_only(reason: str) -> None:
    """Release the base model and enter the cloud-only degraded state.

    Owns the COMPLETE transition: model + tokenizer released and nulled,
    ``cloud_only_reason`` set, ``mode`` set to ``"cloud-only"``, subscribers
    notified.  Callers do not set ``mode``.

    Called from the post-load VRAM budget gate (reason='insufficient_vram')
    and the persistent-CUDA-fault crash-loop guard
    (reason='cuda_fault_persistent'); on the latter the reason is in
    ``_PERMANENT_CLOUD_ONLY_REASONS`` so the GPU is never auto-reclaimed
    (re-entering a poisoned context would re-trigger the fault).

    Every reachable call site is boot-phase code, so no concurrent GPU-lock
    holder can exist; the release therefore runs unlocked.  That precondition
    is checked, not assumed — if a future call site breaks it the release is
    still performed (this is a crash path; refusing would strand the server
    with a resident model it cannot use) and the violation is logged at ERROR.
    """
    from paramem.server.gpu_lock import gpu_lock_is_held

    if gpu_lock_is_held():
        logger.error(
            "_degrade_to_cloud_only(%s): GPU lock is held while degrading — the "
            "unlocked in-process release is only safe from boot-phase callers. "
            "Proceeding (crash path), but this call site needs the lock-aware "
            "teardown used by /gpu/release.",
            reason,
        )
    _release_base_model_in_process()
    _state["cloud_only_reason"] = reason
    _state["model"] = None
    _state["tokenizer"] = None
    _state["mode"] = "cloud-only"
    notify_server(SERVER_CLOUD_ONLY)


def _record_cuda_fatal_exit() -> None:
    """Append the current timestamp to the in-state-dir crash-loop history file.

    Follows the trial_state atomic-write idiom: write to a .pending/ subdir then
    rename into the final path.  Prunes entries older than
    ``config.vram.cuda_fault_history_window_s`` on each write so the file never
    grows unboundedly.

    Best-effort boundary I/O — a filesystem error must NOT prevent os._exit from
    running.  Callers catch all exceptions.
    """
    config = _state.get("config")
    if config is None:
        return
    state_dir = data_state_dir(config.paths.data).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    history_file = state_dir / "cuda_fault_history.json"
    pending_dir = state_dir / ".pending"
    pending_dir.mkdir(exist_ok=True)

    now_ts = time.time()
    window_s: int = config.vram.cuda_fault_history_window_s

    # Read existing history, prune stale entries.
    history: list[float] = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history = [ts for ts in history if (now_ts - ts) < window_s]
    history.append(now_ts)

    payload = json.dumps(history).encode("utf-8")
    pending_file = pending_dir / "cuda_fault_history.json"
    pending_file.write_bytes(payload)
    os.rename(pending_file, history_file)


def _cuda_crashloop_exhausted() -> bool:
    """True when fatal CUDA exits within the history window have reached the burst limit.

    Reads ``<state>/cuda_fault_history.json``; returns False on any read error
    (prefer os._exit retry over wrongly sticking cloud-only on a disk error).
    """
    config = _state.get("config")
    if config is None:
        return False
    state_dir = data_state_dir(config.paths.data).resolve()
    history_file = state_dir / "cuda_fault_history.json"
    if not history_file.exists():
        return False
    try:
        history: list[float] = json.loads(history_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    now_ts = time.time()
    window_s: int = config.vram.cuda_fault_history_window_s
    burst: int = config.vram.cuda_crashloop_burst
    recent = [ts for ts in history if (now_ts - ts) < window_s]
    return len(recent) >= burst


def _cuda_liveness_canary() -> None:
    """Force the CUDA context to surface a latent sticky fault.

    Runs an UNGUARDED torch.cuda.synchronize() so a poisoned context raises
    before the 'ready' log is emitted.  Unlike safe_empty_cache (paramem.utils.vram_guard),
    which swallows synchronize failures, this propagates the error so the
    lifespan fail-fast handler can act on it BEFORE advertising server-ready.
    No-op when CUDA is unavailable or no model is loaded.
    """
    if not torch.cuda.is_available() or _state.get("model") is None:
        return
    torch.cuda.synchronize()  # UNGUARDED — must raise on a sticky context


def _fail_fast_cuda(exc: BaseException, phase: str) -> None:
    """Record the fatal CUDA exit and os._exit(1), or degrade cloud-only if exhausted.

    Recovery is os._exit(1) ONLY (systemd Restart=on-failure → fresh CUDA
    context) UNLESS the crash-loop guard says retries are exhausted, in
    which case this function delegates to :func:`_degrade_to_cloud_only`
    (which DOES call ``_release_base_model_in_process`` — that call is
    accepted there because a poisoned context left in a permanently
    degraded server is worse than one more safe_empty_cache → synchronize
    hit) and the server continues serving cloud-only instead of restarting.
    """
    logger.critical("FATAL CUDA fault during %s — %s", phase, exc)
    if _cuda_crashloop_exhausted():
        logger.critical(
            "CUDA crash-loop guard: burst exhausted — degrading to persistent cloud-only "
            "instead of restarting (reason='cuda_fault_persistent')"
        )
        _degrade_to_cloud_only("cuda_fault_persistent")
        return  # stays up, cloud-only, ~0 GiB
    try:
        _record_cuda_fatal_exit()
    except Exception:
        logger.exception("_fail_fast_cuda: could not record fault history (best-effort)")
    logger.critical(
        "FATAL CUDA fault — calling os._exit(1) for a fresh context via systemd restart"
    )
    os._exit(1)  # systemd Restart=on-failure → fresh CUDA context


def _check_token_ratio_drift(config) -> None:
    """Boot-time token-estimate ratio drift check — the live consumer of
    ``consolidation.extraction_token_estimate_ratio`` (see
    :func:`paramem.utils.tokens.check_ratio_drift`'s docstring for why the
    key would otherwise be inert).  Re-measures the words->tokens fallback
    ratio against the LIVE tokenizer over synthetic samples; a non-``None``
    result means a base-model swap shifted the ratio and the configured
    fallback would under-budget a payload of that shape.  No-op when no
    tokenizer has loaded (cloud-only boot).

    Extracted out of the ``lifespan`` body as a named, unit-testable
    function rather than an inline block with a lifespan-frame local: the
    ``@asynccontextmanager`` lifespan stays suspended at its own ``yield``
    for the app's whole lifetime, so any local it holds persists that
    long too (see this module's BASE-MODEL HOLDER invariant note for the
    general hazard) — a plain function call has no such lifetime and
    leaves nothing behind once it returns.

    Args:
        config: The just-loaded :class:`~paramem.server.config.ServerConfig`
            (the lifespan's own local, not necessarily yet mirrored into
            ``_state["config"]`` at the point this is called).

    Side effects:
        On drift, logs a WARNING and sets
        ``_state["token_ratio_drift_warning"] = {"configured_ratio":
        float, "observed_ratio": float}`` — the dict
        :func:`~paramem.server.attention._collect_token_ratio_drift_items`
        reads to surface the persistent ``/status.attention`` item.
    """
    tokenizer = _state.get("tokenizer")
    if tokenizer is None:
        return
    observed_ratio = check_ratio_drift(
        tokenizer, config.consolidation.extraction_token_estimate_ratio
    )
    if observed_ratio is None:
        return
    logger.warning(
        "Token-estimate ratio drift: configured %.2f tok/word, live tokenizer "
        "observed %.2f tok/word — the estimate_tokens() fallback may "
        "under-budget a payload of the dominant shape. Re-measure and update "
        "consolidation.extraction_token_estimate_ratio.",
        config.consolidation.extraction_token_estimate_ratio,
        observed_ratio,
    )
    _state["token_ratio_drift_warning"] = {
        "configured_ratio": config.consolidation.extraction_token_estimate_ratio,
        "observed_ratio": observed_ratio,
    }


# GPU-lock timeout for lifespan shutdown's base-model release: long enough for
# a background-trainer worker mid-fold to reach its next epoch boundary and
# release (the shutdown flag set earlier in this same teardown only stops
# training AT that boundary, not immediately), short enough that a genuinely
# stuck holder does not indefinitely stall shutdown — the alternative is
# systemd SIGKILL skipping the remaining teardown below the release. Mirrors
# _apply_config_live's _APPLY_CONFIG_LOCK_TIMEOUT_S.
_SHUTDOWN_GPU_LOCK_TIMEOUT_S: float = 60.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    config = _state["config"]

    # Slot hygiene: a second lifespan in the same process (TestClient reuse,
    # or an in-process restart across separate event loops) must never await
    # a task handle left behind by a previous lifespan's shutdown.
    # Shutdown's ``.cancel()`` calls are not awaited (see the shutdown block
    # below), so the done-callback that clears these slots may not have run
    # yet when a new lifespan begins — and a task created on one event loop
    # cannot even be awaited from another. ``base_swap_task`` is the one
    # actually awaited by ``_run_boot_completion_tasks``, so a stale handle
    # there would raise ``CancelledError``/``RuntimeError`` outside that
    # function's own ``except Exception`` isolation; ``boot_completion_task``
    # is unconditionally overwritten later in this same lifespan and is
    # never awaited elsewhere, so it carries no such hazard, but is reset
    # here too for the same clean-slate guarantee.
    _state["base_swap_task"] = None
    _state["boot_completion_task"] = None
    # ``mode`` is process-global and this lifespan is never guaranteed to be
    # the first one in the process (TestClient reuse, in-process restart) —
    # a prior lifespan's degrade can otherwise leave "cloud-only" resident
    # and pin THIS boot's mode-write guard (below) into treating a healthy
    # local boot as an in-progress degrade. Reset to the pre-boot default
    # from the module-level ``_state`` initializer above ("local") so this
    # lifespan starts from the same clean slate as a fresh process.
    _state["mode"] = "local"

    # Deployment-integrity gate: fail loudly if the shared prompt assets are
    # missing (broken checkout / non-editable pip install). Prompts are not
    # packaged; a repo checkout always has them. Runs before anything expensive.
    from paramem.graph.prompts import ensure_prompt_assets

    ensure_prompt_assets(prompts_dir=config.prompts_dir)

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
    )
    # Boot-time reconciliation: a crash mid-checkpoint-save leaves plaintext
    # files inside a partial checkpoint-*/ dir while durable stores are
    # age-encrypted, which would otherwise trip _assert_mode's mixed-state
    # refusal below. Purge those partial dirs (and any dangling
    # staging_resume.json pointer into them) before the gate runs. Gated on
    # _daily_ok (belt-and-suspenders — purge_partial_checkpoints self-gates
    # too via _security_on()).
    if _daily_ok:
        from paramem.training.trainer import purge_partial_checkpoints

        _purged = purge_partial_checkpoints(Path(config.paths.data) / "adapters")
        if _purged:
            logger.warning(
                "Boot reconciliation purged %d partial checkpoint dir(s): %s",
                len(_purged),
                ", ".join(str(p) for p in _purged),
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
    # when the PWA slice is active (or a prior mint already wrote the store
    # file) so a default deployment (mobile_pwa.enabled=false, no prior
    # mint) leaves the store None and the middleware OFF.
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
            Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
        )
        state_dir = data_state_dir(config.paths.data)
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
        # This mirrors the per-kind sweep in _validate_adapter_slot for
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
            "No auto-reclaim for this reason; reclaim manually with "
            "`pstatus --acquire` (POST /gpu/acquire) once the GPU is free."
        )
        notify_server(SERVER_CLOUD_ONLY)
        cloud_only = True
        _state["cloud_only_reason"] = "gpu_conflict"

    # Permanent cloud-only: the GPU pair will NEVER be loaded this process
    # lifetime. explicit/gpu_conflict/cuda_fault_persistent → no auto-reclaim.
    # training (--defer-model) → auto-reclaim will load the GPU pair; reserve
    # the bytes ahead of time. cuda_fault_persistent is included so a sticky
    # CUDA context fault that lands the server cloud-only is never auto-reclaimed
    # (re-entering the poisoned context would re-trigger the same fault).
    permanent_cloud_only = _state.get("cloud_only_reason") in _PERMANENT_CLOUD_ONLY_REASONS

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
            #
            # Cold-tier creation (paramem.models.loader.ensure_resident_tiers)
            # is a VRAM allocation like any other boot-time load, so it must
            # sit behind the same degrade-not-crash posture as the drain wait
            # above and check_post_load_budget below: a VRAM shortfall here
            # produces a running cloud-only server, never a dead unit. Catch
            # ONLY VramExhausted and the fatal-CUDA path — a ConfigStoreMismatch
            # raised inside _load_model_into_state (see
            # paramem.server.config_store_validator.check_config_against_store:
            # a populated interim ring left behind by a disabled episodic tier,
            # or a disabled tier whose registry still holds active keys) must
            # abort the boot loudly, not be swallowed into a silent cloud-only
            # degrade.
            try:
                _load_model_into_state(config)
            except VramExhausted:
                logger.warning(
                    "Boot model load: VRAM exhausted during tier creation — degrading to cloud-only"
                )
                _degrade_to_cloud_only("insufficient_vram")
                cloud_only = True
            except BaseException as _load_exc:
                if is_fatal_cuda_fault(_load_exc):
                    _fail_fast_cuda(_load_exc, "load_model_into_state")
                    cloud_only = True  # on the exhausted-burst path: _degrade ran, stays up
                else:
                    raise

    # Config-derived component construction — single shared routine called by
    # BOTH the lifespan (here) and the live-apply path.  At boot the session
    # buffer is always rebuilt (rebuild_session_buffer=True, the default).
    # Note: _apply_config_in_progress is not set here (boot path); the re-probe gate
    # inside the routine treats a None/absent store as cold and runs the probe.
    try:
        _build_runtime_components(config, cloud_only=cloud_only)
    except BaseException as _bcs_exc:
        if is_fatal_cuda_fault(_bcs_exc):
            _fail_fast_cuda(_bcs_exc, "preload")
            cloud_only = True  # on the exhausted-burst path: _degrade ran, stays up
        else:
            raise

    # Post-load authoritative gate. Runs AFTER _build_runtime_components so
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
            _degrade_to_cloud_only("insufficient_vram")
            cloud_only = True
        elif base_pred is not None:
            delta_mib = (actual_bytes - base_pred) / (1024 * 1024)
            logger.info(
                "VRAM calibration drift: predicted %.2f GiB, measured %.2f GiB (delta %+.0f MiB)",
                base_pred / 2**30,
                actual_bytes / 2**30,
                delta_mib,
            )

    # Boot-time token-estimate ratio drift check — see
    # _check_token_ratio_drift's own docstring.  Runs only when a
    # tokenizer actually loaded (cloud-only boot has none) — that guard
    # lives inside the function, not here.
    _check_token_ratio_drift(config)

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
                # Provider, not an eager snapshot: _build_runtime_components
                # rebinds _state["speaker_store"] on every full config-apply
                # (app.py step 2), and this Wyoming socket is bound once here
                # at lifespan boot — an eager _state.get("speaker_store") would
                # keep gating embedding computation on the pre-apply (possibly
                # stale) store for the life of the process. Mirrors the TTS
                # socket's speaker_store_provider below.
                speaker_store_provider=lambda: _state.get("speaker_store"),
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
                # Provider, not an eager snapshot: _build_runtime_components
                # rebinds _state["speaker_store"] on every full config-apply
                # (app.py step 2), and this Wyoming socket is bound once here
                # at lifespan boot — an eager _state.get("speaker_store") would
                # keep resolving against the pre-apply (possibly stale) store
                # for the life of the process. Mirrors tts_manager_provider.
                speaker_store_provider=lambda: _state.get("speaker_store"),
            )
            logger.info("Wyoming TTS server listening on port %d", config.tts.port)

    # Log the configured cadence for correlation with early boot logs.  The
    # INTERIM cadence (= refresh_cadence, e.g. "12h") drives POST
    # /scheduled-tick at every interim boundary; the tick handler decides
    # whether a given tick is an interim train or a full fold via
    # _is_full_cycle_due.  The full period (refresh_cadence ×
    # max_interim_count, e.g. 84h) is derived for logging and for the
    # deadline backstop in _is_full_cycle_due.
    #
    # Actual systemd timer reconciliation (both the consolidation and backup
    # timers) happens off the event loop in the boot-completion task
    # (_run_boot_completion_tasks, scheduled below) rather than here — a
    # blocking subprocess.run in this pre-yield path would stall uvicorn's
    # bind and the Wyoming STT/TTS listeners behind it.
    interim_cadence = config.consolidation.refresh_cadence or ""
    full_period = config.consolidation.consolidation_period_string
    logger.info(
        "Consolidation cadence — interim cadence %s, max_interim_count=%d, "
        "derived full-consolidation period=%s",
        interim_cadence or "<disabled>",
        config.consolidation.max_interim_count,
        full_period or "<manual only>",
    )

    # _degrade_to_cloud_only commits mode="cloud-only" itself; the boot-computed
    # value must never overwrite it.
    if _state.get("mode") != "cloud-only":
        _state["mode"] = "cloud-only" if cloud_only else "local"
    _state["event_loop"] = asyncio.get_running_loop()

    # Auto-reclaim: only when started without the model (--defer-model).
    # In local mode we already have the GPU — nothing to reclaim. Also never
    # armed when the boot itself already landed on a permanent cloud-only
    # reason (e.g. 'cuda_fault_persistent' from the crash-loop guard above,
    # via _fail_fast_cuda -> _degrade_to_cloud_only): _auto_reclaim_loop does
    # not consult _PERMANENT_CLOUD_ONLY_REASONS itself, so arming it here
    # would reload the base model straight back into the same poisoned CUDA
    # context that just forced the degrade.
    if (
        cloud_only
        and not _state.get("cloud_only_startup", False)
        and _state.get("cloud_only_reason") not in _PERMANENT_CLOUD_ONLY_REASONS
    ):
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
    # and every runtime component is built, launch the orchestration
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
    # Ensure the local-mode adapter directory exists before the first root-level
    # write. Boot-time readers (iter_interim_dirs, registry preload) already
    # tolerate a missing directory; this guarantees it for writes.
    if not cloud_only:
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

    _bs_resume = _state.pop("_base_swap_resume_marker", None)
    if _bs_resume is not None and not cloud_only:
        _bs_live_cfg = (
            Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
        )
        _bs_state_dir = data_state_dir(config.paths.data).resolve()
        _bs_backups_root = (config.paths.data / "backups").resolve()
        # Handle stored in _state (mirroring reclaim_task/config_drift_task
        # above) so it is awaitable (the boot-completion task awaits it
        # before running its own catch-up work) and cancellable at shutdown;
        # an unstored asyncio.create_task(...) result is a GC hazard (the
        # event loop only holds a weak reference). Cleared by its own
        # done-callback once the orchestration finishes, guarded so a newer
        # launch that already replaced the slot is never clobbered.
        _state["base_swap_task"] = asyncio.create_task(
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
        _state["base_swap_task"].add_done_callback(
            functools.partial(_clear_state_task, "base_swap_task")
        )
        logger.info(
            "base-swap resume launched: base_swap_phase=%s old=%s new=%s",
            _bs_resume.base_swap_phase,
            _bs_resume.old_model,
            _bs_resume.new_model,
        )

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

    # Post-preload liveness canary: run an UNGUARDED synchronize() to surface
    # any latent sticky CUDA context fault BEFORE advertising server-ready.
    # safe_empty_cache (above) swallows synchronize failures; this does not.
    try:
        _cuda_liveness_canary()
    except BaseException as _canary_exc:
        if is_fatal_cuda_fault(_canary_exc):
            # On the exhausted-burst path _degrade_to_cloud_only already
            # committed mode="cloud-only"; the server stays up, cloud-only.
            _fail_fast_cuda(_canary_exc, "post-preload canary")
        else:
            raise

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
    _n_user_tokens = _posture_store.count_active() if _posture_store is not None else 0
    log_startup_posture(
        n_user_tokens=_n_user_tokens,
        per_user_active=_posture_store is not None,
    )

    # Boot-completion catch-up: timer reconciliation + missed-schedule
    # backup/consolidation dispatch. Scheduled here (still pre-yield) but
    # deliberately NOT awaited — uvicorn only binds the port after this
    # generator's yield, and a Persistent=true systemd tick that fires
    # during reconcile has nowhere to land until then, so this work runs in
    # the background rather than delaying server-ready. Runs from a
    # module-level coroutine (_run_boot_completion_tasks), not a closure
    # over this frame's locals, so it holds no reference to the base model,
    # the tokenizer, or any other lifespan-frame local (see the
    # BASE-MODEL HOLDER invariant in _release_base_model_in_process).
    _state["boot_completion_task"] = asyncio.create_task(_run_boot_completion_tasks())
    _state["boot_completion_task"].add_done_callback(
        functools.partial(_clear_state_task, "boot_completion_task")
    )

    yield

    # Shutdown — data-safety-first order:
    # 1. Signal training to stop (so in-flight cycles begin winding down).
    # 2. Persist disk-only state (snapshot + speaker flush) before any GPU op —
    #    if a SIGKILL arrives during the slow GPU release, both persistence ops
    #    have already completed.
    # 3. Release the base model (single owner: _release_base_model_in_process).
    # 4. Wyoming / STT / TTS / HA teardown.
    # 5. Final allocator mop-up.
    _shutdown_t0 = time.perf_counter()

    # Signal training to stop at the next epoch boundary so _release_base_model_in_process
    # (via bt.release() join) waits as short as possible.
    consolidation_loop = _state.get("consolidation_loop")
    if consolidation_loop is not None:
        consolidation_loop.shutdown_requested = True
        logger.info("Shutdown flag set — training will stop after current epoch")
    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        bg_trainer._shutdown_requested = True
        bg_trainer._is_training = False
        logger.info("Background trainer stopped")

    # Persist in-memory session state before any GPU op so a SIGKILL-during-release
    # does not drop unconsolidated conversations.
    buffer = _state.get("session_buffer")
    if buffer:
        _t = time.perf_counter()
        buffer.save_snapshot()
        logger.info("shutdown timing: buffer.save_snapshot %.2fs", time.perf_counter() - _t)

    # Flush deferred speaker profile writes — disk-only, must run before GPU release.
    store = _state.get("speaker_store")
    if store:
        _t = time.perf_counter()
        try:
            store.flush()
        except Exception:
            logger.exception("Failed to flush speaker store during shutdown")
        logger.info("shutdown timing: store.flush %.2fs", time.perf_counter() - _t)

    if _state.get("reclaim_task"):
        _state["reclaim_task"].cancel()
    if _state.get("config_drift_task"):
        _state["config_drift_task"].cancel()
    if _state.get("base_swap_task"):
        _state["base_swap_task"].cancel()
    if _state.get("boot_completion_task"):
        _state["boot_completion_task"].cancel()

    # Release the base model — single owner for base-model + bt/loop + intent-handle
    # release. Shutdown does not separately call unload_model on _state["model"];
    # _release_base_model_in_process is the only release path here.
    #
    # A background-trainer worker thread can legitimately still hold
    # gpu_lock_sync mid-fold at this point (the shutdown flag set above only
    # stops training at the NEXT epoch boundary) — release under the lock
    # with a bounded wait rather than racing it unlocked. This cannot
    # deadlock: async gpu_lock holders release on the event-loop thread, but
    # by this point the loop's lock-taking tasks are cancelled just above and
    # in-flight HTTP requests have already drained, so the realistic holder
    # class is sync worker threads, which release on their own thread. On
    # timeout, log and proceed unlocked — shutdown must complete; the
    # alternative is systemd SIGKILL skipping the remaining teardown below.
    from paramem.server.gpu_lock import gpu_lock_sync

    _t = time.perf_counter()
    try:
        with gpu_lock_sync(timeout=_SHUTDOWN_GPU_LOCK_TIMEOUT_S):
            _release_base_model_in_process()
    except TimeoutError:
        logger.error(
            "shutdown: could not acquire GPU lock within %ss — a lock holder "
            "outlasted the shutdown wait; releasing the base model unlocked "
            "so shutdown can complete",
            _SHUTDOWN_GPU_LOCK_TIMEOUT_S,
        )
        _release_base_model_in_process()
    logger.info("shutdown timing: _release_base_model_in_process %.2fs", time.perf_counter() - _t)

    wyoming_server = _state.get("wyoming_server")
    if wyoming_server is not None:
        _t = time.perf_counter()
        wyoming_server.stop()
        logger.info("shutdown timing: wyoming_server.stop %.2fs", time.perf_counter() - _t)
    wyoming_tts = _state.get("wyoming_tts_server")
    if wyoming_tts is not None:
        _t = time.perf_counter()
        wyoming_tts.stop()
        logger.info("shutdown timing: wyoming_tts.stop %.2fs", time.perf_counter() - _t)
    stt = _state.get("stt")
    if stt is not None:
        _t = time.perf_counter()
        stt.unload()
        logger.info("shutdown timing: stt.unload %.2fs", time.perf_counter() - _t)
    tts_manager = _state.get("tts_manager")
    if tts_manager is not None:
        _t = time.perf_counter()
        tts_manager.unload_all()
        logger.info("shutdown timing: tts_manager.unload_all %.2fs", time.perf_counter() - _t)
    if _state.get("ha_client"):
        _t = time.perf_counter()
        _state["ha_client"].close()
        logger.info("shutdown timing: ha_client.close %.2fs", time.perf_counter() - _t)

    # Final mop-up — covers cuBLAS workspaces / allocator slack the per-component
    # unloads couldn't reach while their frame-locals were still live. Without
    # this, the next paramem boot's _gpu_has_compute_processes() sees a [Not Found]
    # ghost PID and routes into permanent gpu_conflict (no auto-reclaim).
    # Note: _release_base_model_in_process already calls safe_empty_cache internally;
    # this second call is still required because STT/TTS unload after the release.
    _t = time.perf_counter()
    safe_empty_cache()
    logger.info("shutdown timing: safe_empty_cache %.2fs", time.perf_counter() - _t)
    _total = time.perf_counter() - _shutdown_t0
    logger.info("shutdown timing: total lifespan teardown %.2fs", _total)


def _clear_state_task(key: str, task: "asyncio.Task") -> None:
    """``asyncio.Task`` done-callback: clear ``_state[key]`` when it still holds *task*.

    Shared by every one-shot background task this module stores a handle
    for (``base_swap_task``, ``boot_completion_task``) so a finished task's
    slot does not keep pointing at a dead ``Task`` object forever. Guards on
    identity (``_state.get(key) is task``) rather than unconditionally
    clearing, so a done-callback firing after a newer task has already
    replaced the slot never clobbers that newer task's handle.
    """
    if _state.get(key) is task:
        _state[key] = None


def _reconcile_scheduling_timers(config) -> None:
    """Reconcile both systemd user timers (consolidation tick, scheduled backup) against *config*.

    Single call site for timer reconciliation, shared by the boot-completion
    task (:func:`_run_boot_completion_tasks`, dispatched off the event loop
    via ``asyncio.to_thread``) and a live config apply (``_apply_config_live``,
    which already runs in an executor thread) — a schedule edit to either
    ``consolidation.refresh_cadence`` or ``security.backups.schedule`` reaches
    systemd from every call site that applies config, not only server boot.

    Each timer's reconcile is independently guarded — a failure in one (a
    malformed schedule string, a ``systemctl`` error) is logged and does not
    block the other.

    Blocking under the hood: both ``systemd_timer.reconcile`` and
    ``backup_timer.reconcile`` share ``systemd_timer._reconcile_timer``,
    which shells out via ``subprocess.run``. Callers on the event loop must
    dispatch this through ``asyncio.to_thread``/an executor — never call it
    directly from an ``async def``.

    Args:
        config: The ``ServerConfig`` to read ``consolidation.refresh_cadence``
            and ``security.backups.schedule`` from.
    """
    from paramem.backup import timer as backup_timer
    from paramem.server import systemd_timer

    interim_cadence = config.consolidation.refresh_cadence or ""
    try:
        msg = systemd_timer.reconcile(interim_cadence)
        logger.info("%s", msg)
    except Exception:
        logger.exception("Failed to reconcile consolidation timer — continuing without schedule")

    backup_schedule = config.security.backups.schedule or ""
    try:
        backup_msg = backup_timer.reconcile(backup_schedule, python_path=sys.executable)
        logger.info("%s", backup_msg)
    except Exception:
        logger.exception("Failed to reconcile backup timer — continuing without scheduled backups")


async def _run_boot_completion_tasks() -> None:
    """Run boot-completion catch-up work once the lifespan has finished setting up server state.

    Runs as a background task created in ``lifespan`` (handle in
    ``_state["boot_completion_task"]``, cancelled at shutdown alongside its
    task siblings). A module-level coroutine, not a closure over the
    lifespan frame — it reads ``_state`` fresh at execution time and holds
    no reference to the base model, the tokenizer, or any lifespan-frame
    local (the BASE-MODEL HOLDER invariant — see
    ``_release_base_model_in_process``).

    Exists because ``systemd``'s ``Persistent=true`` catch-up ticks fire into
    the boot window before uvicorn binds the port (bind happens only after
    this lifespan's ``yield``) and are lost — the timer's own curl has no
    retry, and systemd stamps the catch-up as delivered on trigger, not on
    HTTP success. The server owns catch-up itself instead: at boot
    completion it evaluates dueness directly and dispatches through the
    existing doors (the extracted backup seam, and the same in-process
    consolidation dispatch ``POST /scheduled-tick`` uses).

    In order — each step isolated in its own ``try/except``, so a failure in
    one (logged via ``logger.exception``) never prevents the remaining,
    independent steps from running:

    1. Await ``_state["base_swap_task"]`` (a base-swap crash-recovery
       orchestration launched earlier in this same lifespan, if any) so the
       catch-up work below never races an in-flight config/model swap.
    2. Backup catch-up, evaluated and — when due — run to completion BEFORE
       the consolidation catch-up: a fold rewrites the tier adapter
       directories the snapshot bundle reads, so running the backup after a
       fold would capture the fold's own output as though it predated the
       fold.
    3. Consolidation catch-up via the identical in-process ``AUTO`` dispatch
       ``POST /scheduled-tick`` uses, dispatched only when a read-only peek
       at the durable cadence stamp (:func:`~paramem.server.schedule_grammar.scheduled_run_due`
       against :func:`~paramem.server.schedule_state.read_last_scheduled_run`
       — the identical predicate and stamp the arbitrator itself reads, not
       a second dueness implementation) says ``DUE``. ``_dispatch_consolidation``
       owns every dueness/guard/trial decision below that point; this task
       duplicates none of them. A ``NO_STAMP`` peek is left for the
       arbitrator's own seed-and-noop on the next real tick rather than
       seeded here, so there is exactly one seeding owner. Dispatching
       unconditionally would run the arbitrator's side-effecting pre-stages —
       retroactive orphan-session claim,
       :func:`~paramem.server.consolidation.retire_unattributable_sessions`
       retiring unattributable pending sessions regardless of TTL, and the
       ``pending_rehydration`` migration branch seizing the GPU — on every
       boot with a real cadence configured, whether or not a tick was
       actually missed.
    4. Reconcile both systemd user timers off the event loop, LAST —
       :func:`_reconcile_scheduling_timers` shells out via
       ``subprocess.run`` per timer, dispatched through ``asyncio.to_thread``
       so neither it nor a ``Persistent`` catch-up tick systemd fires as a
       side effect of enabling the unit ever stalls the Wyoming STT/TTS
       listeners. Running this last (rather than first) means both catch-ups
       above have already stamped/completed by the time a reconcile-triggered
       ``Persistent=true`` tick could land — that tick then curls a live
       server and reads not-due, rather than racing the boot task's own
       catch-up work into a double run.

    Because step 1 blocks every later step on the base-swap task, an
    in-flight base-swap resume at boot delays step 4's timer reconcile (and
    steps 2-3) for the resume's full duration. This is deliberate: ``config``
    is read fresh from ``_state`` immediately after step 1 completes, and a
    base-swap resume replaces that config — reconciling (or dispatching
    catch-up) before the swap finishes would act on the pre-swap config.
    """
    _bst = _state.get("base_swap_task")
    if _bst is not None:
        try:
            await _bst
        except Exception:
            logger.exception("Boot catch-up: base-swap task failed — continuing with catch-up")

    config = _state.get("config")
    if config is None:
        return

    from paramem.server.schedule_grammar import (
        ScheduleDueStatus,
        parse_schedule_atom,
        scheduled_run_due,
    )

    # --- Backup catch-up (must run BEFORE the consolidation catch-up). ---
    try:
        backup_schedule = config.security.backups.schedule or ""
        _backup_atom = parse_schedule_atom(backup_schedule)
        if _backup_atom is not None and _backup_atom.kind != "off":
            from paramem.backup.state import last_attempt_epoch as _last_attempt_epoch

            state_dir = data_state_dir(config.paths.data).resolve()
            _last_stamp = _last_attempt_epoch(state_dir)
            # NO_STAMP -> RUN here (the opposite of consolidation's seed-and-noop
            # below): a first backup is cheap and welcome, so its absence must
            # not be faked into looking already-run — mirrors the same policy
            # in backup/__main__.py's own standalone catch-up gate.
            _due_status = scheduled_run_due(backup_schedule, _last_stamp)
            if _due_status in (ScheduleDueStatus.DUE, ScheduleDueStatus.NO_STAMP):
                logger.info(
                    "Boot catch-up: scheduled backup is due (status=%s, schedule=%r) — "
                    "running tier=daily",
                    _due_status.value,
                    backup_schedule,
                )
                # kinds mirrors what the paramem-backup systemd timer's runner
                # itself delegates with (paramem/backup/__main__.py) — the
                # operator's configured artifact list, not a hardcoded default.
                _kinds = list(config.security.backups.artifacts)
                await asyncio.to_thread(_create_backup, _kinds, "daily", None)
            else:
                logger.info(
                    "Boot catch-up: scheduled backup not due (schedule=%r)", backup_schedule
                )
    except Exception:
        logger.exception("Boot catch-up: backup step failed — continuing with remaining steps")

    # --- Consolidation catch-up. ---
    # Only when refresh_cadence is a real schedule, AND only when a read-only
    # peek at the durable cadence stamp says DUE — see the docstring's step 3.
    try:
        cadence = config.consolidation.refresh_cadence or ""
        _cadence_atom = parse_schedule_atom(cadence)
        if _cadence_atom is not None and _cadence_atom.kind != "off":
            from paramem.server import schedule_state as _schedule_state

            _last_scheduled = _schedule_state.read_last_scheduled_run(
                data_state_dir(config.paths.data)
            )
            _cadence_due = scheduled_run_due(cadence, _last_scheduled)
            if _cadence_due is ScheduleDueStatus.DUE:
                status, action = _dispatch_consolidation(ConsolidationAction.AUTO)
                logger.info(
                    "Boot catch-up: consolidation AUTO dispatch — status=%s action=%s",
                    status,
                    action.value,
                )
            else:
                logger.info(
                    "Boot catch-up: consolidation not due (status=%s, cadence=%r) — skipping",
                    _cadence_due.value,
                    cadence,
                )
    except Exception:
        logger.exception(
            "Boot catch-up: consolidation dispatch step failed — continuing with remaining steps"
        )

    # --- Timer reconcile — LAST (see docstring step 4). ---
    try:
        await asyncio.to_thread(_reconcile_scheduling_timers, config)
    except Exception:
        logger.exception("Boot catch-up: timer reconcile step failed")


app = FastAPI(title="ParaMem", version="0.1.0", lifespan=lifespan)

# Bearer-token auth on all REST endpoints whenever a UserTokenStore is
# wired (see _build_user_token_store).  No-op when no store is wired (loud
# WARN emitted from lifespan after the store decision is made).  There is
# no separate shared-token credential any more — every accepted token is a
# UserTokenStore entry, attributed or not.  PARAMEM_API_TOKEN survives only
# as the carrier env var infra consumers (the systemd scheduling timer, the
# HA custom component) read to source the Authorization header value for a
# token that must itself be minted into the store (see DEPLOYMENT.md — Per-
# user token management).
from paramem.server.auth import (  # noqa: E402
    BearerTokenMiddleware,
    log_startup_posture,
)

app.add_middleware(
    BearerTokenMiddleware,
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
    denied.  In auth-OFF mode (no user-token store wired) the server is
    open for use — ``BearerTokenMiddleware`` stamps the same non-admin
    ``scope == "chat"`` on every pass-through request (see ``auth.py`` OFF
    branch), so admin endpoints 403 here until a per-user store is
    configured (fail-closed admin), while unguarded endpoints (chat/voice)
    remain reachable without a credential.

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

    # Authenticated speaker from an ATTRIBUTED per-user bearer token (set by
    # BearerTokenMiddleware on ON-per-user requests).  None for an
    # unattributed per-user token (no speaker attribution) and unauthenticated
    # mode.
    auth_speaker_id: str | None = getattr(http_request.state, "speaker_id", None)

    # Forced routing — bypass normal routing for direct provider testing.
    # Supports: "ha", "cloud", "cloud:anthropic", "cloud:openai", "cloud:google"
    if request.route and request.route.startswith(("ha", "cloud")):
        _speaker_id, speaker = _resolve_speaker(
            request, buffer, _state.get("speaker_store"), auth_speaker_id=auth_speaker_id
        )
        # Same typed boundary decision as the normal /chat path — forced
        # routing selects the PROVIDER, it does not buy a history-egress
        # bypass: a speakerless caller gets no history on the forced route
        # either.
        _serving = ServingPath.for_speaker(_speaker_id)
        _forced_history = (
            []
            if _serving is ServingPath.RELAY
            else buffer.get_conversation_turns(request.conversation_id)
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
        elif request.route.startswith("cloud"):
            parts = request.route.split(":", 1)
            agent = (
                _state.get("cloud_providers", {}).get(parts[1])
                if len(parts) == 2
                else _state.get("cloud_agent")
            )
            if agent is not None and _state["mode"] == "cloud-only":
                # Cloud-only: no local model, so no ParaMem-held knowledge and
                # no anonymizer — the same plain-cloud-agent posture as
                # _relay_route.  Routed through the one egress funnel
                # (model/tokenizer=None selects its cannot-anonymize branch);
                # history is still drop-gated there.
                _forced_cloud_permitted = (
                    _state.get("cloud_only_reason") not in _INVOLUNTARY_CLOUD_ONLY_REASONS
                    or _state["config"].cloud.allow_degraded_serving
                )
                result = await loop.run_in_executor(
                    None,
                    lambda: answer_via_cloud(
                        request.text,
                        agent,
                        _state["config"],
                        model=None,
                        tokenizer=None,
                        speaker=speaker,
                        speaker_id=_speaker_id,
                        history=_forced_history,
                        cloud_permitted=_forced_cloud_permitted,
                    ),
                )
            elif agent is not None:
                # Local mode: forced routing selects the PROVIDER, it does not
                # buy a policy bypass.  The turn goes through the one egress
                # funnel, so cloud_mode and the personal verdict apply exactly
                # as they do on the routed path.  answer_via_cloud reaches the
                # live model here (anonymizer's extract_graph/anonymize_turn
                # calls generate() under base_model_inference — see
                # inference.py:answer_via_cloud), so this dispatch needs the
                # same GPU discipline as the routed local path: abort any
                # in-flight background training, then hold the GPU lock for
                # the duration.
                _forced_is_personal = is_self_referential(
                    request.text,
                    personal_referent_config=_state["config"].personal_referent,
                )
                _abort_background_training_for_inference()

                from paramem.server.gpu_lock import gpu_lock

                async with gpu_lock():
                    result = await loop.run_in_executor(
                        None,
                        lambda: answer_via_cloud(
                            request.text,
                            agent,
                            _state["config"],
                            is_personal=_forced_is_personal,
                            model=_state.get("model"),
                            tokenizer=_state.get("tokenizer"),
                            speaker=speaker,
                            speaker_id=_speaker_id,
                            history=_forced_history,
                        ),
                    )
        if result and result.text:
            resolved_text = resolve_speaker_tokens(
                result.text, _state.get("speaker_store"), current_speaker_id=_speaker_id
            )
            return ChatResponse(text=resolved_text, escalated=True, speaker=speaker)
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
    speaker_id, speaker = (
        _resolved.speaker_id,
        _resolved.speaker,
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


class ServingPath(Enum):
    """Typed boundary decision stamped by speaker resolution.

    ``PERSONAL`` — a speaker_id resolved (named or anonymous-promoted); the
    turn is eligible for the full local dispatch (``handle_chat``):
    parametric-memory probing, cloud escalation under the ``is_personal``
    privacy gate, abstention, etc.

    ``RELAY`` — no speaker_id resolved at all.  The turn is served by the
    relay path (``_relay_route``) regardless of server mode: HA / cloud
    only, no conversation history egresses, personal interrogatives get the
    canned no-identity response, and the session is appended with
    ``speaker_id=None`` — consolidation's existing 3-way triage
    (``paramem.server.consolidation.classify_session``) then decides its
    fate with no consolidation-side change needed: a text-only ``/chat``
    RELAY turn carries no voice embedding either, so it classifies
    ``UNIDENTIFIABLE`` (dropped, no TTL); a ``/voice`` RELAY turn DOES carry
    the STT voice embedding even with no resolved speaker, so it classifies
    ``HOLDABLE`` instead (retained pending retro-claim, not dropped) — the
    same embedding-presence rule every other holdable session uses.

    The one constructor, :meth:`for_speaker`, is the sole place this
    decision is made — every caller (the normal ``/chat`` / ``/voice`` path
    and forced routing) derives ``ServingPath`` from it rather than
    re-deriving the ``speaker_id is None`` check locally.
    """

    PERSONAL = "personal"
    RELAY = "relay"

    @classmethod
    def for_speaker(cls, speaker_id: str | None) -> "ServingPath":
        """Return ``PERSONAL`` when *speaker_id* resolved, else ``RELAY``."""
        return cls.PERSONAL if speaker_id is not None else cls.RELAY


@dataclass(frozen=True)
class ResolvedSpeaker:
    """Speaker resolution result shared by POST /chat and POST /voice.

    Attributes
    ----------
    speaker_id:
        Canonical speaker identifier (e.g. ``"speaker0"``), or ``None``
        when resolution failed or the caller is fully anonymous.
    speaker:
        Display name returned by the speaker store (used in ChatResponse).
        For an anonymous-promoted speaker this is the raw ``speaker{N}``
        token itself (the store's convention until disclosure) — callers
        that need a human-safe salutation use
        ``store.resolve_speaker_name(speaker_id)`` directly (see the
        greeting-prefix assembly below), not this field.

        There is deliberately no ``serving`` field here: the
        :class:`ServingPath` boundary decision is fully derived from
        ``speaker_id`` (:meth:`ServingPath.for_speaker`), so storing it
        alongside ``speaker_id`` would let the two drift out of sync if a
        caller ever mutated one without the other.  ``_run_chat_turn``
        (the sole consumer) calls ``ServingPath.for_speaker(speaker_id)``
        itself; forced routing (the ``/chat`` handler's ``request.route``
        branch) does the same at its own fork.
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
    caller carries no per-user speaker attribution (``auth_speaker_id is
    None`` — an unattributed per-user token, or auth-OFF).

    Parameters
    ----------
    request:
        Incoming :class:`ChatRequest`.  ``request.speaker_embedding`` is
        used for embedding-based resolution and enrollment when present.
    auth_speaker_id:
        Speaker ID attached by :class:`~paramem.server.auth.BearerTokenMiddleware`
        for an ATTRIBUTED per-user token.  ``None`` for an unattributed
        per-user token or auth-OFF (no store wired).
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
        All resolution outputs; callers destructure as needed.
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

            # Promote to a canonical speaker{N} id so facts flow through
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
    # gone after the server restart that allocated their speaker{N}).
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

    # Check greeting before routing (applies to all paths).  The salutation
    # is looked up directly via ``resolve_speaker_name`` — it returns
    # ``None`` for an anonymous/undisclosed profile (the store's own
    # suppression), which naturally yields a nameless greeting ("Good
    # morning.") without a separate display-name field.  ``speaker_id``
    # itself is never used as a salutation.
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
            greeting_name = store.resolve_speaker_name(speaker_id)
            if greeting_name:
                greeting_prefix = f"{greeting}, {greeting_name}. "
            else:
                greeting_prefix = f"{greeting}. "
            store.confirm_greeting(speaker_id)

    return ResolvedSpeaker(
        speaker_id=speaker_id,
        speaker=speaker,
        follow_up=follow_up,
        greeting_prefix=greeting_prefix,
        effective_language=effective_language,
    )


#: ``_state["cloud_only_reason"]`` values that mean the local model the
#: operator CHOSE to run is unavailable against their wishes.  These are the
#: only states ``cloud.allow_degraded_serving`` gates.  Deliberate cloud-only
#: ("explicit", "released") and transient internal states ("training",
#: "live_reload") are not degraded serving and always proceed.
_INVOLUNTARY_CLOUD_ONLY_REASONS: frozenset[str] = frozenset(
    {
        "gpu_conflict",
        "insufficient_vram",
        "reload_failed",
        "apply_failed",
        "config_refused",
        "cuda_fault_persistent",
    }
)

#: Cloud-only reasons that must never be auto-reclaimed this process lifetime.
_PERMANENT_CLOUD_ONLY_REASONS: frozenset[str] = frozenset(
    {"explicit", "gpu_conflict", "cuda_fault_persistent"}
)

#: Prepended once per conversation when a turn is served over the degraded
#: cloud path.  App-layer prefix (same mechanism as the greeting) — never
#: written to the session buffer, so it can never reach a training transcript.
_DEGRADED_SERVING_NOTICE = "My local memory is offline right now, so a cloud model is answering. "

#: Prepended once per conversation when a turn is served over the relay path
#: because no speaker could be resolved at all (``ServingPath.RELAY`` with no
#: server-wide cloud-only condition).  Same app-layer-only, never-persisted
#: mechanism as :data:`_DEGRADED_SERVING_NOTICE` — see :func:`_notice_once`.
_SPEAKERLESS_RELAY_NOTICE = (
    "I don't recognize who's speaking, so I'm answering without your personal memory. "
)


def _notice_once(conversation_id: str, notice_kind: str, notice: str) -> str:
    """Return *notice* the first time requested for (*conversation_id*, *notice_kind*), else "".

    Shared registry backing both :data:`_DEGRADED_SERVING_NOTICE`
    (``notice_kind="degraded"``) and :data:`_SPEAKERLESS_RELAY_NOTICE`
    (``notice_kind="speakerless"``) — one mechanism, one registry
    (``_state["relay_notice_conversations"]``), keyed by the
    ``(conversation_id, notice_kind)`` pair.  Keying on the pair — not on
    *conversation_id* alone — means a conversation that already announced
    one notice kind still announces the OTHER kind once, the first time it
    applies; only a REPEAT of the same kind in the same conversation is
    suppressed.

    Parameters
    ----------
    conversation_id:
        The conversation this notice would be attached to.
    notice_kind:
        Short discriminator for which notice this is (``"degraded"`` or
        ``"speakerless"``) — part of the registry key alongside
        *conversation_id*.
    notice:
        The notice text to gate, or ``""`` when no notice applies this turn
        (returned unchanged without touching the registry).

    Returns
    -------
    str
        *notice* on the first call for this (*conversation_id*,
        *notice_kind*) pair; ``""`` on every later call for the same pair
        (or when *notice* was already empty).
    """
    if not notice:
        return ""
    announced: set[tuple[str, str]] = _state["relay_notice_conversations"]
    key = (conversation_id, notice_kind)
    if key in announced:
        return ""
    announced.add(key)
    return notice


def _abort_background_training_for_inference() -> None:
    """Abort in-flight background training so an inference turn can acquire the GPU.

    ``abort_for_inference()`` sets the per-job abort flag and waits up to
    ``consolidation.abort_quiesce_timeout_s`` for training to stop at the
    next step boundary and release the GPU lock; force-stops the trainer
    (bypassing the graceful wait) if it does not abort in time.  No-op when
    no training is running.  Called OUTSIDE ``async with gpu_lock()`` so the
    caller's subsequent lock acquisition succeeds without contention.

    Shared by every inference call site in this module that follows with
    ``async with gpu_lock()``: the local ``handle_chat`` dispatch and the
    relay leg (``_relay_route``) in :func:`_run_chat_turn`.
    """
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


async def _run_chat_turn(
    *,
    text: str,
    conversation_id: str,
    speaker_id: str | None,
    speaker: str | None,
    speaker_embedding: list[float] | None,
    language: str | None,
    greeting_prefix: str | None,
) -> tuple[ChatResult, str]:
    """Execute a single conversation turn (shared by POST /chat and POST /voice).

    Encapsulates the post-speaker-resolution orchestration that is identical
    for text and voice turns: the scheduler debounce stamps, training-abort,
    relay vs local routing, session buffer appends, scheduled-training
    enqueue, and greeting prefix application.

    Both ``/chat`` and ``/voice`` callers are responsible for resolving
    *greeting_prefix* before calling this function.  The relay-vs-local
    ``ServingPath`` decision is made HERE, from *speaker_id*, via the one
    constructor (:meth:`ServingPath.for_speaker`) — it is not threaded in
    by the caller, so it can never drift from the ``speaker_id`` actually
    passed (the same constructor forced routing calls independently at its
    own fork, in the ``/chat`` handler).

    Conversation history is server-authoritative: it is read from
    ``SessionBuffer.get_conversation_turns(conversation_id)`` BEFORE this turn's
    own ``buffer.append`` calls run, so it never doubles-up the current user
    utterance — the read always happens first, and it is the only read of
    history in this function.

    Parameters
    ----------
    text:
        The user's message text (already transcribed for voice turns).
    conversation_id:
        Conversation / session identifier.
    speaker_id:
        Resolved canonical speaker ID, or ``None`` for a speaker who could
        not be resolved at all — drives the ``ServingPath.for_speaker``
        decision below (``None`` → ``RELAY``).
    speaker:
        Display name of the speaker (resolved by ``_resolve_speaker``), or
        ``None``.
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
        :class:`~paramem.server.inference.ChatResult` (token-space, as
        persisted) and *spoken_text* is *result.text* with every
        ``speaker{N}`` token resolved to a display name (or the third-party
        descriptor) via :func:`~paramem.server.speaker.resolve_speaker_tokens`,
        with *greeting_prefix* prepended AFTER that resolution.
    """
    buffer = _state["session_buffer"]
    speaker_store = _state.get("speaker_store")

    # THE typed boundary decision, derived from speaker_id alone — see the
    # docstring above.  ``PERSONAL`` dispatches to the local ``handle_chat``
    # path (speaker_id guaranteed non-None there).  ``RELAY`` dispatches to
    # ``_relay_route`` with no history egress and ``identity_absent=True``,
    # regardless of server mode.
    serving = ServingPath.for_speaker(speaker_id)

    # Read BEFORE any append below — the current turn is not yet in the
    # buffer, so this can never include it.
    history = buffer.get_conversation_turns(conversation_id)

    # Debounce stamps — monotonic for scheduler, wall-clock for /status display.
    # Both writes run on the asyncio event-loop thread (cooperative scheduling),
    # so no lock is needed.
    _state["last_chat_time"] = datetime.now(timezone.utc)
    _state["last_chat_monotonic"] = time.monotonic()

    # Relay fork: the EXISTING cloud-only chain serves two distinct
    # conditions through the one leg —
    #   1. server-wide cloud-only mode (no local model loaded at all), or
    #   2. this particular request carries no resolved speaker at all
    #      (``serving is ServingPath.RELAY``), regardless of server mode.
    # Condition 2 never egresses history and never touches parametric
    # memory; condition 1 is the pre-existing degraded-serving behavior.
    # A request can be both at once — the degraded notice takes priority
    # over the speakerless notice in that case (see the notice selection
    # below), but either alone routes here.
    server_cloud_only = _state["mode"] == "cloud-only"
    identity_absent = serving is ServingPath.RELAY
    if server_cloud_only or identity_absent:
        # Degraded serving: the local model is gone for a reason the operator
        # did not choose.  The CLOUD leg is closed unless they opted in; the
        # HA leg stays open either way — HA carries no ParaMem-held knowledge
        # and runs on the user's own network, so breaking it during a GPU
        # conflict buys no privacy.  Speakerless-only (server otherwise
        # healthy) is not degraded — the cloud leg is fully open.
        degraded = server_cloud_only and (
            _state.get("cloud_only_reason") in _INVOLUNTARY_CLOUD_ONLY_REASONS
        )
        cloud_permitted = (not degraded) or _state["config"].cloud.allow_degraded_serving

        # The relay leg now runs classifier/encoder calls
        # (``_is_personal_interrogative`` / ``is_self_referential``) and, in
        # local mode, live-model calls (anonymize + base-model fallback) —
        # the same GPU-touching work the local ``handle_chat`` dispatch
        # below does.  Abort background training and hold the GPU lock for
        # the duration, mirroring the local leg exactly (see
        # ``_abort_background_training_for_inference``'s docstring); a
        # cloud-only server (model is None) still takes the lock, which is
        # cheap and harmless with no GPU work behind it.
        _abort_background_training_for_inference()

        from paramem.server.gpu_lock import gpu_lock

        async with gpu_lock():
            loop = asyncio.get_running_loop()
            result: ChatResult = await loop.run_in_executor(
                None,
                lambda: _relay_route(
                    text=text,
                    # No history egress for a speakerless request, even when the
                    # server is ALSO cloud-only — identity_absent is the stronger
                    # privacy condition.  Full history (still drop-gated inside
                    # answer_via_cloud) for a server-wide cloud-only turn from a
                    # resolved speaker.
                    history=([] if identity_absent else history),
                    config=_state["config"],
                    cloud_permitted=cloud_permitted,
                    ha_client=_state.get("ha_client"),
                    cloud_agent=_state.get("cloud_agent"),
                    language=language,
                    speaker_id=speaker_id,
                    identity_absent=identity_absent,
                    # Live model/tokenizer in local mode (identity_absent
                    # turn on an otherwise-healthy server) so the relay's
                    # cloud leg can sanitize via the local anonymizer and
                    # the final fallback can reach the local base model;
                    # genuinely None in server-wide cloud-only mode.
                    model=_state.get("model"),
                    tokenizer=_state.get("tokenizer"),
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
        cloud_text = result.text
        buffer.append(
            conversation_id,
            "assistant",
            cloud_text,
            speaker_id=speaker_id,
            speaker=speaker,
        )
        # Notice selection — once per conversation via the shared
        # ``_notice_once`` registry.  Degraded-serving takes priority when
        # both conditions apply (server-wide state is the stronger signal).
        # Prefix order: greeting, then notice, then the resolved answer —
        # resolution happens AFTER persist (cloud_text above is what was
        # written to the buffer), the greeting stays an app-layer prepend.
        if degraded and cloud_permitted:
            notice = _notice_once(conversation_id, "degraded", _DEGRADED_SERVING_NOTICE)
        elif identity_absent:
            notice = _notice_once(conversation_id, "speakerless", _SPEAKERLESS_RELAY_NOTICE)
        else:
            notice = ""
        resolved_text = resolve_speaker_tokens(
            cloud_text, speaker_store, current_speaker_id=speaker_id
        )
        spoken_text = f"{greeting_prefix or ''}{notice}{resolved_text}"
        return result, spoken_text

    # Local mode — normal inference with entity routing.
    # Abort background training if active, then acquire the GPU lock.
    # abort_for_inference() sets the per-job abort flag and waits up to 30 s
    # for training to stop at the next step boundary and release the GPU lock.
    # _active_quiesced is set OUTSIDE gpu_lock_sync so the caller's
    # async with gpu_lock() below succeeds without lock contention.
    _abort_background_training_for_inference()

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=text,
                conversation_id=conversation_id,
                speaker=speaker,
                speaker_id=speaker_id,
                history=history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=_state["config"],
                router=_state["router"],
                cloud_agent=_state.get("cloud_agent"),
                ha_client=_state.get("ha_client"),
                language=language,
                # Active-store migration override: when a mode-switch is in
                # progress or interrupted, the inference path falls back to
                # the source mode's store. None == use config.consolidation.mode.
                effective_mode=_state.get("effective_mode"),
                memory_store=_state["memory_store"],
            ),
        )

    # Persist BEFORE resolving — the buffer keeps the token-space turn
    # (speaker{N}), never the display name.  Resolution happens only at the
    # spoken_text boundary below, after this append.
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

    resolved_text = resolve_speaker_tokens(
        response_text, speaker_store, current_speaker_id=speaker_id
    )
    spoken_text = f"{greeting_prefix}{resolved_text}" if greeting_prefix else resolved_text
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
        enrollment "What's your name?" message on the unattributed-caller
        path — an unattributed per-user token, or auth-OFF).  ``None``
        when no follow-up is needed.
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

    - **Unattributed token / no attributed identity** (``auth_speaker_id is
      None`` — an unattributed per-user token minted via ``mint-user-token
      --unattributed``, or auth-OFF mode): the voice embedding IS computed
      and passed through :func:`_resolve_and_enroll_speaker` — the same
      enrollment/greeting/name-disclosure path as ``POST /chat``.  A fresh
      per-utterance ``conversation_id`` is generated on each request (one
      POST = one push-to-talk press).  The session-buffer retro-claim
      propagates identity across conversation_ids by embedding so two-turn
      enrollment still works.

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
    # carries an attributed per-user token, identity is authoritative and no
    # embedding is needed (cheap path).  An unattributed token (or auth-OFF
    # mode) stamps no speaker_id (``auth_speaker_id is None`` either way),
    # so the voice embedding is required for identification and enrollment.
    auth_speaker_id: str | None = getattr(http_request.state, "speaker_id", None)

    # Compute embedding only when no speaker was attributed by the token;
    # skip when an attributed per-user token already gives identity (saves
    # CPU cost and avoids unnecessary WeSpeaker inference).
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

    # Per-utterance transport/enrollment id on the unattributed-caller path
    # (unattributed per-user token, or auth-OFF): each push-to-talk press
    # is an independent POST /voice call, so a fresh id feeds
    # _resolve_and_enroll_speaker's unknown-speaker grouping (keyed by this
    # id) and the LLM enrollment trigger. The retro-claim in
    # session_buffer.claim_sessions_for_speaker works across these ids by
    # embedding, so two-turn enrollment still works. This id is NOT the
    # session-buffer conversation_key used below — that is derived from the
    # RESOLVED speaker after enrollment runs. On the ATTRIBUTED per-user
    # path, a stable id allows multi-turn context.
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

    # Session-buffer conversation_key: derived from the RESOLVED speaker so
    # all voice turns from the same speaker (identified, anonymous, or
    # bearer-token) share one session-grouping + idle-rotation lineage in
    # SessionBuffer — no per-utterance session, no "voice-default" merge of
    # distinct speakers. A turn with no resolved speaker_id (no embedding, or
    # register_anonymous failed) is genuinely un-attributable; fall back to
    # the transport handle, same rule the text path uses.
    buffer_conversation_key = (
        f"voice-{_resolved.speaker_id}" if _resolved.speaker_id is not None else conversation_id
    )

    # Reconcile _open routing state onto the SINGLE key voice actually
    # buffers under. _resolve_and_enroll_speaker (shared with /chat) wrote
    # speaker state via set_speaker(conversation_id, ...) — the transport
    # id, correct for its own enrollment/grouping purposes but distinct from
    # buffer_conversation_key. Re-record the resolved speaker under the real
    # buffer key and drop the now-redundant transport-id entry so it can't
    # orphan (it never gets a session_id, so retirement-based pruning can't
    # reach it — one leaked entry per unattributed-caller utterance otherwise).
    if buffer_conversation_key != conversation_id:
        buffer.set_speaker(buffer_conversation_key, _resolved.speaker_id, _resolved.speaker or "")
        buffer.discard_open_routing_key(conversation_id)

    # Route through the shared turn orchestrator.  Use the resolved effective
    # language (Whisper → stored preference → None) for routing and TTS.
    result, spoken_text = await _run_chat_turn(
        text=text,
        conversation_id=buffer_conversation_key,
        speaker_id=_resolved.speaker_id,
        speaker=_resolved.speaker,
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

    The subscription is persisted under the speaker_id bound to an
    ATTRIBUTED per-user bearer token (set by
    :class:`~paramem.server.auth.BearerTokenMiddleware`).  An unattributed
    per-user token or an unauthenticated request is rejected with HTTP 403.
    The endpoint is deduplicated per speaker — re-subscribing the same
    endpoint is a no-op.

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
        When no per-user speaker_id is attached to the request (an
        unattributed per-user token, or unauthenticated).
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

    # Require an ATTRIBUTED per-user token — an unattributed token does not
    # bind to a speaker_id.
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
            speaker_name = speaker_store.resolve_speaker_name(auth_speaker_id)
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


def _derive_consolidation_status_fields(
    state_dir: Path,
) -> tuple[dict | None, dict | None, dict | None]:
    """Derive ``last_consolidation_error``, ``last_consolidation_result``, and
    the durable calibration run row from disk.

    Reads the incident store (``incidents.json``) and the run-status registry
    (``run_status.json``) once and returns the three ``/status`` fields.  All
    sources are read at build time on every ``/status`` poll — no RAM snapshot
    survives.

    Returns
    -------
    tuple[dict | None, dict | None, dict | None]
        ``(last_consolidation_error, last_consolidation_result, calibration_result)``

        ``last_consolidation_error``:
            The ``detail`` dict of the most-recent **active** incident whose
            type is one of the consolidation/vram/extraction failure families, or
            ``None`` when none are active.  Shape for ``vram_exhausted`` matches
            the historic ``{"type": "vram_exhausted", "phase": str, "at": iso8601}``
            so the ``StatusResponse`` field stays HTTP-stable.

        ``last_consolidation_result``:
            The ``RunRecord.to_dict()`` for op_type ``"consolidation"`` from
            ``run_status.json``, or ``None`` when no run has been recorded.

        ``calibration_result``:
            The ``RunRecord.to_dict()`` for op_type ``"calibration"`` from
            ``run_status.json``, or ``None``.  Durable across a restart —
            ``StatusResponse.calibration_run``'s fallback for when
            ``_state["calibration_run"]`` (the live slot, reset on every
            process start) has never been written this process lifetime.
    """
    # --- last_consolidation_error: derived from active incidents ---
    consolidation_error: dict | None = None
    _consolidation_incident_types = frozenset(
        {
            "vram_exhausted",
            "extraction_failed",
            "consolidation_crash",
            "training_crash",
            "migration_phase_failed",
            "migration_error",
            "full_consolidation_overdue",
            "interim_cap_reached",
            "interim_overflow_pending",
        }
    )
    try:
        incidents = read_incidents(state_dir)
        # Most-recent active incident from the relevant families.
        active = [
            i for i in incidents if i.status == "active" and i.type in _consolidation_incident_types
        ]
        if active:
            # last_seen is an ISO string; lexicographic comparison of UTC ISO strings
            # gives correct chronological ordering.
            most_recent = max(active, key=lambda i: i.last_seen)
            consolidation_error = most_recent.detail
    except Exception:
        logger.exception("_derive_consolidation_status_fields: could not read incidents")

    # --- last_consolidation_result / calibration_result: derived from run_status.json ---
    consolidation_result: dict | None = None
    calibration_result: dict | None = None
    try:
        last_runs = read_last_runs(state_dir)
        rec = last_runs.get("consolidation")
        if rec is not None:
            consolidation_result = rec.to_dict()
        cal_rec = last_runs.get("calibration")
        if cal_rec is not None:
            calibration_result = cal_rec.to_dict()
    except Exception:
        logger.exception("_derive_consolidation_status_fields: could not read run_status")

    return consolidation_error, consolidation_result, calibration_result


@app.get("/status", response_model=StatusResponse)
async def status():
    """Server health and state."""
    config = _state["config"]
    model = _state["model"]

    # The episodic tier carries trained weights — not merely that an
    # adapter object exists (residency does not imply readiness). A
    # resident-but-cold episodic tier (created but never trained) reads
    # False here. Read from the recorded snapshot
    # (_record_tier_weight_state, written at every adapter-mutation
    # boundary — mount and go-live promote) rather than measured here:
    # has_prior_trained_weights walks named_parameters(), and /status is
    # polled roughly once per second during a fold.
    adapter_loaded = _state.get("tier_weight_state", {}).get("episodic", False)

    # Adapter inventory: enumerate configured kinds + interim capacity. Main
    # adapters contribute 1 each when their tier exists; interim contributes
    # max_interim_count (the capacity ceiling enforced by the VRAM validator).
    _status_tier_configs = config.tier_config_map()
    adapter_config_counts: dict[str, int] = {kind: 1 for kind in _status_tier_configs}
    if config.consolidation.max_interim_count > 0:
        adapter_config_counts["interim"] = config.consolidation.max_interim_count

    # Currently active adapter on the live PeftModel. PEFT only keeps one
    # adapter active at a time (set_adapter / switch_adapter). None when the
    # model hasn't loaded (cloud-only) or no adapters exist yet.
    active_adapter: str | None = None
    if model is not None and hasattr(model, "active_adapter"):
        active_adapter = active_adapter_name(model)

    # Active key count comes from the one authoritative MemoryStore — the same
    # store /debug/dump and the recall path read.  It is loaded from disk (main
    # + interim tiers) at boot via load_registries_from_disk.  When no live
    # store is constructed (e.g. cloud-only before preload), count from disk
    # using the store's own loader rather than re-inlining a registry scan.
    store = _state.get("memory_store")
    if store is not None:
        keys_count = len(store.all_active_keys())
        tier_key_counts = {
            tier: len(store.active_keys_in_tier(tier)) for tier in store.tiers_with_registry()
        }
    else:
        # No live store yet (cloud-only before preload) — read each tier's
        # registry directly via the one shape predicate (KeyRegistry.load)
        # instead of a fresh MemoryStore.load_registries_from_disk, which
        # aborts the WHOLE read on the first tier that fails the shape
        # check.  One foreign-shaped tier registry must not 500 the whole
        # endpoint; skip that tier and keep counting the rest.  Tier
        # "unreadable" is already owned end-to-end by
        # _record_unverified_tier_incidents (surfaced via the attention
        # block) — this cold path only needs an honest count, not a second
        # reporter, so a skip is logged and otherwise silent here.
        from paramem.memory.interim_adapter import iter_tier_roots
        from paramem.training.key_registry import KeyRegistry

        tier_key_counts = {}
        for tier, tier_root in iter_tier_roots(config.adapter_dir):
            reg_path = tier_root / "indexed_key_registry.json"
            try:
                tier_key_counts[tier] = len(KeyRegistry.load(reg_path).list_active())
            except ValueError:
                logger.warning(
                    "/status: %s is not a KeyRegistry-shaped registry file — "
                    "excluding tier %s from keys_count",
                    reg_path,
                    tier,
                )
        keys_count = sum(tier_key_counts.values())

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

    # Next interim bucket boundary: stamps are floored to the
    # refresh_cadence boundary measured from midnight, so the next
    # boundary is fully deterministic from the clock. None when cadence is
    # disabled (manual-only mode).
    from paramem.server.schedule_grammar import compute_schedule_period_seconds

    next_interim_seconds: int | None = None
    _refresh_seconds = compute_schedule_period_seconds(config.consolidation.refresh_cadence)
    if _refresh_seconds and _refresh_seconds > 0:
        _now = datetime.now()
        _midnight = _now.replace(hour=0, minute=0, second=0, microsecond=0)
        _since_mid = int((_now - _midnight).total_seconds())
        _next_boundary = ((_since_mid // _refresh_seconds) + 1) * _refresh_seconds
        next_interim_seconds = max(0, _next_boundary - _since_mid)

    # Honest next-full prediction — gate-derived, not a raw timer tick.
    next_full_consolidation_seconds = _seconds_until_next_full_consolidation(config)
    # Oldest un-folded interim stamp for the renderer inline math.
    oldest_interim_stamp = _oldest_interim_stamp(config)

    # Per-component VRAM ledger — device-wide totals from torch.cuda.mem_get_info
    # plus the component deltas measured at load time.
    # CUDA-guarded: on unavailable/fault log and leave fields None.
    vram_used_mib: int | None = None
    vram_total_mib: int | None = None
    if torch.cuda.is_available():
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info()
            vram_total_mib = _total_bytes >> 20
            vram_used_mib = (_total_bytes - _free_bytes) >> 20
        except Exception:  # noqa: BLE001
            logger.warning("/status: torch.cuda.mem_get_info() failed — VRAM fields omitted")
    _vram_comps_bytes: dict[str, int] = _state.get("vram_components") or {}
    vram_components: dict[str, int] = {k: v >> 20 for k, v in _vram_comps_bytes.items() if v > 0}
    vram_paramem_mib: int | None = sum(vram_components.values()) if vram_components else None

    # Background trainer
    bt = _state.get("background_trainer")
    bg_active = bool(bt and getattr(bt, "is_training", False))
    bg_adapter = None  # surfaced in /status; resolved by the active training caller now

    # Thermal-throttle / quiet-hours snapshot. Read from the loaded config so the
    # block is present even before the BackgroundTrainer has been constructed.
    # ``currently_throttling`` reflects the policy gate only; the actual throttle
    # additionally requires ``training_temp_limit > 0`` and temp above limit —
    # surfaced here is "would the policy allow throttling right now".
    from paramem.training.thermal_throttle import is_thermal_policy_active

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
    episodic_rank = (
        _status_tier_configs["episodic"].rank if "episodic" in _status_tier_configs else None
    )

    # Per-kind adapter spec — one row per tier in tier_config_map() (disabled
    # tiers are absent from the map, so pstatus never displays a row for
    # one). target_kind compresses target_modules into a category label —
    # "attn+mlp" means MLP layers are in the set, else "attn". A caller
    # interested in the exact list can hit the yaml.
    def _target_kind(target_modules: list[str]) -> str:
        for t in target_modules or []:
            tl = t.lower()
            if "mlp" in tl or "gate" in tl or "up_proj" in tl or "down_proj" in tl:
                return "attn+mlp"
        return "attn"

    adapter_specs: dict[str, dict] = {
        _kind: {
            "rank": _cfg.rank,
            "alpha": _cfg.alpha,
            "learning_rate": _cfg.learning_rate,
            "target_kind": _target_kind(_cfg.target_modules),
        }
        for _kind, _cfg in _status_tier_configs.items()
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
            _sm = read_trial_marker(data_state_dir(config.paths.data).resolve())
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
        from paramem.server.schedule_grammar import (
            compute_schedule_period_seconds,
            parse_schedule_atom,
        )

        _backups_root = (config.paths.data / "backups").resolve()
        _state_dir = data_state_dir(config.paths.data).resolve()

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
        _backup_timer_state = systemd_timer.cached_timer_state("paramem-backup", max_age_seconds=5)
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
            # compute_schedule_period_seconds raises on unparseable input;
            # guard with parse_schedule_atom(...) is None rather than
            # try/except (_schedule_str is operator-supplied config, already
            # excluded from off/empty above, but may still be malformed).
            _interval_s = (
                compute_schedule_period_seconds(_schedule_str)
                if parse_schedule_atom(_schedule_str) is not None
                else 0
            )
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

    # Derive last_consolidation_error, last_consolidation_result, and the
    # durable calibration run row from durable stores (incidents.json +
    # run_status.json) rather than from RAM.  All three are computed once
    # per /status poll; no RAM snapshot survives across restarts.
    _status_state_dir = data_state_dir(config.paths.data).resolve()
    (
        _consolidation_error,
        _consolidation_result,
        _calibration_result,
    ) = _derive_consolidation_status_fields(_status_state_dir)

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
        last_consolidation_error=_consolidation_error,
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
        last_consolidation_result=_consolidation_result,
        pending_enrollments=len(_state.get("pending_enrollments") or []),
        scheduler_started=scheduler_active,
        adapter_manifest=_state.get("adapter_manifest_status", {}),
        config_drift=_state.get("config_drift", {}),
        attention=attention_block,
        migration=migration_block,
        backup=backup_block,
        hold=hold_block,
        encryption=_state.get("encryption", "off"),
        server_started_at=_state.get("server_started_at", ""),
        pending_documents=pending_documents,
        pending_transcripts=pending_transcripts,
        pending_rehydration=bool(_state.get("pending_rehydration", False)),
        effective_mode=_state.get("effective_mode"),
        last_reclaim_error=_state.get("last_reclaim_error"),
        vram_used_mib=vram_used_mib,
        vram_total_mib=vram_total_mib,
        vram_paramem_mib=vram_paramem_mib,
        vram_components=vram_components,
        next_full_consolidation_seconds=next_full_consolidation_seconds,
        tier_key_counts=tier_key_counts,
        oldest_interim_stamp=oldest_interim_stamp,
        store_quarantined=_state.get("store_quarantine"),
        calibration_run=_state.get("calibration_run") or _calibration_result,
    )


@app.get("/integrity", response_model=IntegrityResponse, dependencies=[Depends(require_admin)])
async def integrity_check():
    """Run the infrastructure integrity check and return the report.

    Cloud-only-safe — no GPU or model dependency.  Verifies every tier's
    ``indexed_key_registry.json`` (which now carries the unified simhash map)
    and ``key_metadata.json``, plus common (whole-store) files
    (``speaker_profiles.json``, ``observed_languages.json``,
    ``state/backup.json``).  A keyed tier's live slot and its registry↔slot
    binding verdict are resolved venue-blind via
    :func:`~paramem.adapters.registry_binding.verify_tier_binding` — one
    ``"manifest"``-category row and one ``"payload"``-category row per tier,
    both from the SAME resolution; the payload row reads the BOUND slot's
    ``graph.json`` when its manifest declares a ``"simulate"`` payload
    (nothing writes a tier-root ``graph.json`` any more).

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

    Recovery matches state certainty. When the reload primitive HANDLES the
    failure internally (returns a reason instead of raising) the process is
    known-clean — released to ~0 GiB VRAM, mode already cloud-only — so the
    server just stays cloud-only and reports the reason in both
    ``cloud_only_reason`` and a reason-specific flag; no restart is
    triggered. Insufficient free VRAM (an external GPU consumer holds the
    device) gets its own flag, ``deferred_insufficient_vram: true`` — a
    restart there would only crash-loop on the lifespan VRAM budget gate.
    A config that contradicts the store on disk (the residual race between
    an earlier validation and this reload) reports
    ``cloud_only_reason: "config_refused"`` with ``reload_failed: false`` —
    a refused reload is not a failed one. Any other handled failure reports
    ``reload_failed: true``. Only an escaped exception (unknown process
    state) falls back to ``_restart_service`` (``will_restart: true``).

    Refuses (without touching hold state) while a consolidation cycle is
    in flight (503 ``consolidating`` — same idiom as ``/gpu/release``) or
    while a base-swap migration is actively running (409
    ``base_swap_active`` — same idiom as ``/migration/confirm`` and
    ``/migration/rollback``): reloading the base model out from under
    either would race the GPU-touching work they hold the lock for.
    """
    if _state.get("consolidating", False):
        return JSONResponse(
            status_code=503,
            content={
                "error": "consolidating",
                "detail": (
                    "GPU acquire refused: a consolidation cycle is in flight. "
                    "Retry once /status reports consolidating=false."
                ),
            },
        )
    if (_state.get("migration") or {}).get("base_swap_active", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "base_swap_active",
                "message": (
                    "A base-swap migration is actively running. "
                    "Wait for it to complete (or fail) before acquiring the GPU."
                ),
            },
        )

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
    reload_failed = False
    will_restart = False
    cloud_only_reason: str | None = None
    if needs_reload:
        try:
            from paramem.server.gpu_lock import gpu_lock

            # lock_held=True: gpu_lock() holds the non-reentrant threading.Lock
            # across run_in_executor, mirroring the auto-reclaim loop — the
            # primitive's internal _set_voice_pipeline_profile calls must not
            # re-acquire it. CRITICAL BOUND: this wraps ONLY the reload
            # dispatch — the insufficient-VRAM branch below also dispatches
            # _set_voice_pipeline_profile("cpu") via executor with the
            # default lock_held=False, and that call acquires the same lock;
            # widening this wrap to cover it would deadlock.
            async with gpu_lock():
                reason = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: _live_reload_base_model(lock_held=True)
                )
        except Exception:  # noqa: BLE001
            # Unknown process state — the primitive did not get a chance to
            # signal a handled outcome. Only this path restarts the service.
            logger.exception(
                "In-process reload failed during /gpu/acquire; falling back to restart"
            )
            will_restart = True
            _restart_service()
        else:
            if reason is None:
                reloaded_live = True
                # Voice drain+restore is now owned by _live_reload_base_model
                # (partial-path success restore runs inside the primitive).
                # ── Base-swap deferred-resume hook ──────────────────────────
                # When a phaseA_done base-swap marker exists and the
                # orchestration is not actively running (base_swap_active=False),
                # the reload that just succeeded means Phase B can now run.
                # Re-launch the orchestration in resume mode so Phase B
                # proceeds automatically without operator intervention.
                _bs_mig = _state.get("migration") or {}
                if not _bs_mig.get("base_swap_active", False):
                    _config_for_resume = _state.get("config")
                    if _config_for_resume is not None:
                        _sd_resume = data_state_dir(_config_for_resume.paths.data).resolve()
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
                                else DEFAULT_SERVER_CONFIG_PATH
                            )
                            # Store the handle in the same slot
                            # _run_boot_completion_tasks awaits and shutdown
                            # cancels — an unstored asyncio.create_task(...)
                            # result is a GC hazard (the event loop only holds
                            # a weak reference). base_swap_active=False
                            # (checked above) means no orchestration is
                            # currently running, but guard the slot explicitly
                            # rather than silently overwriting: a non-None
                            # handle here would mean a just-completed
                            # orchestration's done-callback has not yet
                            # cleared it, and launching a second one while
                            # that race is open is not safe to assume away.
                            if _state.get("base_swap_task") is not None:
                                logger.warning(
                                    "/gpu/acquire: base_swap_task slot already occupied — "
                                    "skipping deferred Phase B re-launch this cycle "
                                    "(base_swap_active=False but a task handle is still "
                                    "present; retry once it clears)"
                                )
                            else:
                                _state["base_swap_task"] = asyncio.create_task(
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
                                _state["base_swap_task"].add_done_callback(
                                    functools.partial(_clear_state_task, "base_swap_task")
                                )
                                logger.info(
                                    "/gpu/acquire: re-launching deferred base-swap Phase B "
                                    "(old=%s new=%s)",
                                    _deferred_marker.old_model,
                                    _deferred_marker.new_model,
                                )
            elif reason == "insufficient_vram":
                # Free device memory cannot hold the model (an external GPU
                # consumer holds it). A restart would only re-hit the
                # lifespan VRAM budget gate and crash-loop, so stay
                # cloud-only and tell the operator to free the GPU first.
                # No voice dispatch needed here: the primitive's entry drain
                # (unconditional, before its own VRAM gate) already leaves
                # voice on CPU on every path that can produce this reason.
                deferred_insufficient_vram = True
                cloud_only_reason = reason
                logger.warning(
                    "/gpu/acquire: insufficient free VRAM to reload the model — "
                    "staying cloud-only. Free the GPU and retry `pstatus --acquire`."
                )
            elif reason == "config_refused":
                # The store changed between an earlier config-promotion door's
                # validation and this reload — the reload's own refusal is the
                # race safety net. A refused reload is not a FAILED one: the
                # primitive already released and recorded an incident, so
                # reload_failed stays False here — reload_failed means the
                # load itself broke, not that it was correctly declined.
                cloud_only_reason = reason
                logger.error(
                    "/gpu/acquire: reload refused — config contradicts the store "
                    "on disk. Fix the config or the store, then retry "
                    "`pstatus --acquire`."
                )
            else:
                # Handled failure ("reload_failed" / "apply_failed"): the
                # primitive already released the partial allocation and left
                # the server cloud-only in a known-clean state. No restart —
                # recovery matches state certainty.
                reload_failed = True
                cloud_only_reason = reason
                logger.error(
                    "/gpu/acquire: in-process reload failed (%s) — staying "
                    "cloud-only. Retry `pstatus --acquire` once the underlying "
                    "issue clears, or restart the service explicitly if the "
                    "failure persists.",
                    reason,
                )
    return {
        "cleared": cleared,
        "was_active": hold_before["hold_active"],
        "owner_pid": hold_before["owner_pid"],
        "owner_alive": hold_before["owner_alive"],
        "will_restart": will_restart,
        "reloaded_live": reloaded_live,
        "deferred_insufficient_vram": deferred_insufficient_vram,
        "reload_failed": reload_failed,
        "cloud_only_reason": cloud_only_reason,
    }


def _build_store_contents(
    config,
    *,
    model,
    tokenizer,
) -> "tuple[dict, dict, dict, dict]":
    """Build fresh store contents entirely off-store, or raise if any tier fails verification.

    Reads registries and bookkeeping from disk, then fills entry content —
    the mirror's boot fill act — by probing the source medium (adapter
    weights or on-disk ``graph.json``, selected by
    ``config.consolidation.mode``), and returns three fresh dicts plus a
    stats dict.  The live store is NOT touched — the caller publishes via
    :meth:`~paramem.memory.store.MemoryStore.swap`.  Every entry entering
    ``new_entries`` is SimHash-verified against the staged fingerprints by
    the source medium's own ``finalize_recalled`` before it ever reaches
    this function.  A shortfall in the fill is telemetry, never a verdict:
    every key admitted before a failure stays cached, and the fill records
    (or clears) a ``preload_recall_incomplete`` incident naming what was
    missed — it never aborts the build and never discards what it already
    has.  A train-venue call with no model resident defers the whole fill
    act (:func:`~paramem.memory.source.train_venue_deferred`) rather than
    attempting one — a simulate-venue call always fills, since
    :class:`~paramem.memory.source.DiskMemorySource` needs no model.

    There is no per-tier half-publish: a single tier whose registry↔slot
    binding is not publishable
    (:attr:`~paramem.adapters.registry_binding.TierBinding.publishable` is
    ``False``) raises
    :class:`~paramem.adapters.registry_binding.TierBindingUnpublishable`
    (via :func:`~paramem.adapters.registry_binding.raise_tier_binding_unpublishable`)
    immediately after the tree walk, before any entry preload or
    bookkeeping read — nothing partial is built. The caller (the boot/lift
    store step, :func:`_hydrate_memory_store_in_place`) catches this
    alongside :class:`~paramem.memory.store.BookkeepingInvariantViolation`
    and quarantines the whole store rather than publishing a subset of
    tiers.

    This is the single canonical builder, used by
    :func:`_hydrate_memory_store_in_place` (boot / in-process reload / the
    lift) — called immediately followed by ``store.swap()`` on success.

    **BASE-MODEL HOLDER INVARIANT** — the ``WeightMemorySource`` is a
    frame-local created and dropped within this function.  The caller passes
    ``model`` and ``tokenizer`` as direct kwarg expressions (never via a
    caller local).  The three returned dicts hold NO model reference.
    Setting ``_source = None`` before return releases the only in-frame
    handle.  A surviving reference here would re-introduce the cloud-only
    VRAM leak fixed 2026-05-21.

    Parameters
    ----------
    config:
        Live server config object.
    model:
        Base model handle — passed directly as a kwarg expression at the call
        site; do NOT bind to a caller local before passing.
    tokenizer:
        Tokenizer handle — same constraint.

    Returns
    -------
    tuple of (new_entries, new_registry, new_bookkeeping, stats)
        ``new_entries``: ``dict[tier, dict[key, entry]]``
        ``new_registry``: ``dict[tier, KeyRegistry]`` — every tier from
            ``stats["tier_bindings"]`` (a raise above already guarantees
            every one is publishable by the time this is built).
        ``new_bookkeeping``: ``dict[key, bookkeeping_record]``
        ``stats``: ``{"preload_complete": bool,
                      "tier_bindings": dict[str, TierBinding],
                      "meta_loaded": int, "meta_orphaned": int}``.
        ``preload_complete`` is ``False`` only when the fill was deferred
        for lack of a resident model on the train venue — never when it ran
        and came up short (that is telemetry, recorded as an incident
        below, not a completeness failure).
        ``tier_bindings`` carries the actual
        :class:`~paramem.adapters.registry_binding.TierBinding` objects
        (not a lossy status-string projection) — a consumer that needs to
        report *why* an unverified tier lost verification (e.g. the
        quarantine incident) reads ``binding.detail`` directly rather than
        re-deriving it.

    Raises
    ------
    ~paramem.adapters.registry_binding.TierBindingUnpublishable
        At least one tier (main or interim) failed registry↔slot
        verification.
    ~paramem.memory.store.BookkeepingInvariantViolation
        A tier's registry has known keys but no ``key_metadata.json``
        covering them.
    """
    from paramem.adapters.registry_binding import (
        raise_tier_binding_unpublishable,
        verify_adapter_tree,
    )
    from paramem.memory.entry import content_only_entry, is_admissible_probe_result
    from paramem.memory.source import build_memory_source as _build_memory_source
    from paramem.memory.source import train_venue_deferred as _train_venue_deferred
    from paramem.memory.store import MemoryStore as _MemoryStoreB

    stats: dict = {
        "preload_complete": True,
        "tier_bindings": {},
        "meta_loaded": 0,
        "meta_orphaned": 0,
    }

    # ------------------------------------------------------------------ #
    # Registry — verify fresh from disk, per tier; no live store          #
    # interaction.  There is no per-tier half-publish: a single           #
    # unpublishable tier fails the WHOLE build, before any entry preload  #
    # or bookkeeping read runs, via                                      #
    # raise_tier_binding_unpublishable — the caller (the boot/lift store  #
    # step) catches it and quarantines the store rather than swapping in #
    # a registry map with one tier silently missing.                     #
    # ------------------------------------------------------------------ #
    new_registry: dict = {}
    stats["tier_bindings"] = verify_adapter_tree(config.adapter_dir)
    _unpublishable_tiers = {
        _tier: _binding
        for _tier, _binding in stats["tier_bindings"].items()
        if not _binding.publishable
    }
    if _unpublishable_tiers:
        for _tier, _binding in _unpublishable_tiers.items():
            logger.error(
                "Tier %s registry binding unverified (%s: %s) — quarantining the store",
                _tier,
                _binding.status,
                _binding.detail,
            )
        raise_tier_binding_unpublishable(stats["tier_bindings"])
    for _tier, _binding in stats["tier_bindings"].items():
        new_registry[_tier] = _binding.registry

    # ------------------------------------------------------------------ #
    # Transient store — hoisted above the entry section for use by       #
    # load_bookkeeping_from_disk further down.  Safe to build this       #
    # early: tier_for_known_key (used there) reads only _registry, so    #
    # building it before the entry section cannot shift meta_loaded /    #
    # meta_orphaned.  NOT the live store singleton.                      #
    # ------------------------------------------------------------------ #
    _tmp_store = _MemoryStoreB()
    for _t, _r in new_registry.items():
        _tmp_store.load_registry(_t, _r)

    # ------------------------------------------------------------------ #
    # Entry content — from the source medium (adapter weights or         #
    # on-disk graph.json, selected by config.consolidation.mode).        #
    # ------------------------------------------------------------------ #
    new_entries: dict = {}

    if not config.inference.preload_cache:
        # Intentional opt-out: the mirror is never filled, so every probe
        # goes straight to the live door (MemoryStore.probe_source) and pays
        # its per-key latency there — the cache stays plain off.
        # This counts as a CLEAN pass for the mirror — there is no fill to be
        # incomplete about — so a stale preload_recall_incomplete incident
        # from an earlier preload_cache=true pass is resolved here rather
        # than left to warn about a mirror that no longer serves.
        resolve_incidents_by_type(data_state_dir(config.paths.data), "preload_recall_incomplete")
        logger.info(
            "preload_cache: disabled — store stays entry-empty; inference pays source latency"
        )
    else:
        # Build a temporary in-memory view of the (published) registry to
        # enumerate active keys.  We cannot use the live store here; build
        # from the fresh registry dict.
        _preload_keys_by_tier: dict[str, list[str]] = {}
        for _tier, _reg in new_registry.items():
            _active = _reg.list_active()
            if _active:
                _preload_keys_by_tier[_tier] = _active

        if not _preload_keys_by_tier:
            # No active keys — nothing to preload; store is correctly empty
            # and, like the preload_cache=false arm above, this is a clean
            # pass for the mirror.
            resolve_incidents_by_type(
                data_state_dir(config.paths.data), "preload_recall_incomplete"
            )
        elif _train_venue_deferred(config.consolidation.mode, model):
            # Train venue, no model resident (cloud-only boot, or a failed
            # load): defer the whole fill act to the next act with a model
            # rather than raising build_memory_source's ValueError.  The
            # completion flag stays False so the caller's re-probe gate
            # re-attempts once a model is resident.  Any existing
            # preload_recall_incomplete incident is left UNTOUCHED here — the
            # fill act itself is deferred, not run, so it has produced no new
            # verdict; the deferred fill's own record-or-clear site (below)
            # runs when it actually executes at the next /gpu/acquire.
            stats["preload_complete"] = False
            logger.info(
                "preload_cache: deferring entry preload — no model resident "
                "(cloud-only mode or model load failed); the fill runs on the "
                "next act with a model, and inference pays source latency "
                "on each query until then"
            )
        else:
            _total = sum(len(v) for v in _preload_keys_by_tier.values())

            # NOTE — this probe's results are NOT routed through a second
            # confidence gate below: both WeightMemorySource and
            # DiskMemorySource gate their own results against the same
            # on-disk fingerprints before returning (a hit below threshold
            # comes back as a failure marker, handled by the miss predicate
            # in the per-tier loop below like any other miss), so a second
            # pass here would be a second invocation of the same
            # transformation (and the boot-path tests drive this with
            # MagicMock registries that carry no fingerprints at all).
            #
            # Mode-aware source, built by the one factory.  Select from
            # config.consolidation.mode — NOT from _state["mode"] (that
            # conflates consolidation persistence mode with runtime mode).
            # BASE-MODEL HOLDER (_source frame-local — set to None before return)
            _source = _build_memory_source(
                mode=config.consolidation.mode,
                adapter_dir=config.adapter_dir,
                batch_size=config.consolidation.recall_probe_batch_size,
                model=model,
                tokenizer=tokenizer,
            )
            _medium_name = type(_source).__name__
            logger.info(
                "preload_cache: probing %d active key(s) across %d tier(s) via %s",
                _total,
                len(_preload_keys_by_tier),
                _medium_name,
            )
            # Pre-task GPU cooldown gate — wait until GPU is cool
            # before the ~198-key generate burst.  Bounded by
            # cooldown_gate_max_wait_boot_s (default 60 s <
            # TimeoutStartSec=120) so boot cannot be SIGKILL-ed.
            # Proceeds with a WARNING on timeout rather than hanging.
            # Sits BEFORE the per-tier probe loop below, whose exception
            # handler classifies a fault via is_fatal_cuda_fault, so the
            # device settles before that fail-fast-guarded burst.
            wait_for_cooldown(
                config.vram.cooldown_gate_threshold_c,
                config.vram.cooldown_gate_max_wait_boot_s,
                config.vram.cooldown_gate_poll_s,
                label="preload",
            )

            # Probe PER TIER (not one grouped call over every tier) so a
            # mid-fill raise cannot discard tiers already decoded: each
            # tier's own try/except boundary lets everything admitted by
            # earlier tiers stay cached, with the remaining (not-yet-probed)
            # tiers counted as missed rather than the whole result replaced.
            # Same GPU cost either way — WeightMemorySource already does one
            # switch_adapter + one batched generate per tier internally, so
            # this only isolates exceptions, not batching.
            _hits = 0
            _missed_by_tier: dict[str, list[str]] = {}
            _probe_failed = False
            for _tier, _keys in _preload_keys_by_tier.items():
                if _probe_failed:
                    # An earlier tier's probe failed (non-fatal) — every
                    # later tier is an unattempted miss, not a re-attempt.
                    _missed_by_tier.setdefault(_tier, []).extend(_keys)
                    continue
                try:
                    _tier_results = _source.probe({_tier: _keys})
                except Exception as _probe_exc:
                    if is_fatal_cuda_fault(_probe_exc):
                        # Sticky context loss — propagate so the lifespan
                        # fail-fast handler os._exit(1)s into a fresh process
                        # (the only recovery); never swallowed into telemetry.
                        logger.critical(
                            "preload_cache: FATAL CUDA context fault during probe "
                            "— context poisoned, process restart required: %s",
                            _probe_exc,
                        )
                        raise
                    logger.exception(
                        "preload_cache: source probe failed for tier %s; everything "
                        "admitted before this failure stays cached, the rest answers "
                        "None at the cache door (see the preload_recall_incomplete "
                        "incident for what was missed)",
                        _tier,
                    )
                    _probe_failed = True
                    _missed_by_tier.setdefault(_tier, []).extend(_keys)
                    continue
                # Project once through content_only_entry, count hits,
                # collect misses.  is_admissible_probe_result covers a
                # malformed source result too (absent, a failure marker —
                # including a confidence-gate drop — or missing one of the
                # four content fields) so it becomes a clean miss instead of
                # a cached empty-string triple.
                for _key in _keys:
                    _entry = _tier_results.get(_key)
                    if not is_admissible_probe_result(_entry):
                        _missed_by_tier.setdefault(_tier, []).append(_key)
                        continue
                    new_entries.setdefault(_tier, {})[_key] = content_only_entry(_entry)
                    _hits += 1
            # Drop the WeightMemorySource frame-local — the preload probe is
            # complete; the source must not outlive this function's frame.
            _source = None

            logger.info(
                "preload_cache: cached %d / %d active key(s) via %s",
                _hits,
                _total,
                _medium_name,
            )
            # The fill ran (whatever it produced): completion is unaffected
            # by a shortfall — a shortfall is telemetry, never a verdict.
            # One record-or-clear site for the preload_recall_incomplete
            # incident, keyed by the venue.
            _state_dir = data_state_dir(config.paths.data)
            if _hits < _total:
                record_incident(
                    _state_dir,
                    type="preload_recall_incomplete",
                    key=config.consolidation.mode,
                    severity="warning",
                    summary=(
                        f"Store preload cached {_hits}/{_total} active key(s) via "
                        f"{_medium_name} — the missed keys answer nothing until the "
                        f"mirror re-warms (next boot/config-apply fill, or the next "
                        f"go-live that rebuilds the tier)"
                    ),
                    detail={
                        "hits": _hits,
                        "total": _total,
                        "missed_by_tier": {
                            tier: keys[:10] for tier, keys in _missed_by_tier.items()
                        },
                        "source": _medium_name,
                    },
                )
                logger.warning(
                    "preload_recall_incomplete: preload_cache could not materialise %d / %d "
                    "active keys via %s — the missed keys answer nothing until the mirror "
                    "re-warms (next boot/config-apply fill, or the next go-live that "
                    "rebuilds the tier)",
                    _total - _hits,
                    _total,
                    _medium_name,
                )
            else:
                resolve_incidents_by_type(_state_dir, "preload_recall_incomplete")

    # ------------------------------------------------------------------ #
    # Bookkeeping — read from every tier's key_metadata.json;            #
    # entry-independent.  Reuses the transient store hoisted above (with #
    # the fresh registries already installed) so load_bookkeeping_from_disk #
    # can use tier_for_known_key() to resolve ownership and skip orphans. #
    # ------------------------------------------------------------------ #
    # No try/except here: a tier whose registry has known keys but no
    # key_metadata.json, or a row set that does not cover every known key,
    # is a violation of the every-known-key-has-a-row invariant and must
    # fail the boot store build loudly (BookkeepingInvariantViolation)
    # rather than swap in a registry with zero rows and continue.
    _meta_stats = _tmp_store.load_bookkeeping_from_disk(config.adapter_dir)
    # Extract the populated _bookkeeping dict from the temp store.
    # iter_bookkeeping snapshots under the lock — safe, no external refs.
    new_bookkeeping = dict(_tmp_store.iter_bookkeeping())
    stats["meta_loaded"] = _meta_stats["loaded"]
    stats["meta_orphaned"] = _meta_stats["orphaned"]
    logger.info(
        "load_bookkeeping_from_disk: loaded=%d orphaned=%d",
        _meta_stats["loaded"],
        _meta_stats["orphaned"],
    )

    return new_entries, new_registry, new_bookkeeping, stats


_TIER_REGISTRY_UNVERIFIED_INCIDENT_TYPE = "tier_registry_unverified"


def _record_or_resolve_tier_health(
    config,
    *,
    tier: str,
    unhealthy_status: "str | None",
    detail: str,
    candidate_count: int,
    resolves_payload_status: bool = True,
) -> None:
    """Record (or clear) ONE tier's ``tier_registry_unverified`` incident.

    THE single record-or-clear implementation for the type — both callers
    that mint a per-tier health verdict compose this rather than writing
    the incident shape a second way:

    * :func:`_record_unverified_tier_incidents` — the post-fold drift-
      detection sweep, passing a fold finalizer's own publish verdict
      (``binding.status``/``binding.detail``/``binding.candidate_count``) —
      a FULL :class:`~paramem.adapters.registry_binding.TierBinding`
      resolution, so its healthy signal (``unhealthy_status=None``) already
      proved the bound slot's payload digest still matches (see
      :func:`~paramem.adapters.registry_binding.verify_tier_binding` step
      7). Leaves *resolves_payload_status* at its default ``True``.
    * :func:`_stale_mark_keys` — an erase door's post-mutation report,
      passing a :class:`~paramem.memory.persistence.RestampResult`'s
      status and reason for a tier left unbound, ``None`` for a rebound
      tier, and ``count_slot_candidates(tier_root)`` for the count. A
      restamp NEVER reads the payload file — it only proves the registry
      now binds a slot by hash — so it passes ``resolves_payload_status=
      False``: its own healthy signal is weaker than the sweep's and must
      not be mistaken for one.

    ``unhealthy_status is None`` resolves any prior incident for *tier*
    (idempotent no-op when there was none) — the tier is bound/verified —
    UNLESS *resolves_payload_status* is ``False`` and the existing incident
    (if any) was last recorded with
    :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`: a
    restamp-only healthy signal proves nothing about the payload, so it
    must not silently clear a payload-level failure the drift sweep
    recorded — the incident stays open until the sweep itself recomputes
    the full binding and finds the payload verified. This is the one
    reconciliation point between the two callers; no second incident type
    is introduced for it. Otherwise a new incident is recorded, with
    *detail* (the actual failure reason) in the payload — the incident
    store is the plaintext control-plane that survives a keyless restart,
    exactly when the ERROR log line the same failure also produced is gone.

    Sticky-payload-status invariant: a payload-level status
    (:data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`) recorded
    by a FULL-authority caller (*resolves_payload_status* ``True``) can be
    OVERWRITTEN or RESOLVED only by another full-authority caller. This
    applies on BOTH paths through this function, not only the resolve path
    above: :func:`~paramem.server.incidents.record_incident` replaces
    ``detail`` wholesale on every call, so a limited-authority RECORD (e.g.
    the erase door's own restamp-only failure, ``unhealthy_status`` set to
    its own — unrelated — status) would otherwise silently erase a
    previously-recorded ``PAYLOAD_MISMATCH`` marker from ``detail["status"]``
    without ever having verified the payload itself; a later limited-
    authority RESOLVE would then find no marker to guard against and clear
    an incident the drift sweep never actually re-verified. The RECORD path
    below therefore looks up any existing open incident the same way the
    RESOLVE path does and, when *resolves_payload_status* is ``False`` and
    that incident's ``detail["status"]`` is already
    :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`, keeps
    ``status`` pinned to :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`
    in the freshly-recorded ``detail`` — the caller's own *detail* text
    (which every current caller already folds its own status string into,
    e.g. ``_stale_mark_keys``'s ``f"... manifest re-stamp {result.status}"``)
    still lands verbatim in ``detail["detail"]``, so nothing about the NEW
    failure is lost — only the row's headline ``status`` field stays pinned
    to the unresolved payload-level marker. One incident type, no second
    field, no sidecar marker.

    Severity follows the same primary/secondary tiering every manifest row
    this type mints uses (:func:`_is_primary_adapter`): ``"failed"`` for
    the primary (episodic) tier, ``"info"`` otherwise — a transient interim
    slot or a secondary-tier (semantic/procedural) failure must not raise a
    failed-level row.

    The recorded detail/message deliberately names ``POST /backup/restore``
    (restore the affected tier from a snapshot bundle, then restart) and
    ``GET /integrity`` as the operator's exit — never ``/reconsolidate``,
    which cannot rebuild a tier whose registry itself is unverified.

    Args:
        config: Live server config object; only ``paths.data`` is read.
        tier: The tier (or interim slot) name this verdict is for.
        unhealthy_status: ``None`` when *tier* is bound/verified (resolves
            any prior incident, subject to *resolves_payload_status*);
            otherwise the failure status string
            (:data:`~paramem.adapters.registry_binding.KEY_COUNT_MISMATCH`,
            :data:`~paramem.adapters.registry_binding.KEYS_WITHOUT_SLOT`,
            and :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`
            each get distinct wording; every other value a generic one).
        detail: The failure detail to record verbatim in the incident
            payload. Ignored when *unhealthy_status* is ``None``.
        candidate_count: The number of on-disk slot candidates for *tier*
            at the time of this verdict. Ignored when *unhealthy_status*
            is ``None``.
        resolves_payload_status: Whether THIS caller's healthy signal is
            authoritative enough to clear a prior
            :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH`
            record. ``True`` (default) for a caller that recomputed the
            full :class:`~paramem.adapters.registry_binding.TierBinding`;
            ``False`` for a restamp-only caller. Ignored when
            *unhealthy_status* is not ``None``.
    """
    from paramem.adapters.registry_binding import (
        KEY_COUNT_MISMATCH,
        KEYS_WITHOUT_SLOT,
        PAYLOAD_MISMATCH,
    )

    state_dir = data_state_dir(config.paths.data)

    def _is_sticky_payload_mismatch(existing_row: "dict | None") -> bool:
        # Evaluated under the incidents-store lock, against the row as it
        # stands at write time — never a pre-lock read, which would race a
        # concurrent writer between the read and this call and could lose a
        # payload-mismatch marker the drift sweep is relying on staying put.
        return (
            existing_row is not None
            and existing_row.get("detail", {}).get("status") == PAYLOAD_MISMATCH
        )

    if unhealthy_status is None:
        if resolves_payload_status:
            resolve_incident(state_dir, _TIER_REGISTRY_UNVERIFIED_INCIDENT_TYPE, tier)
        else:
            # A restamp-only caller never reads the payload — only the
            # drift sweep's own next full-binding pass can clear a sticky
            # PAYLOAD_MISMATCH marker. skip_if vetoes the resolve under the
            # same lock that reads the row, so no separate pre-lock read is
            # needed to make that decision.
            resolve_incident(
                state_dir,
                _TIER_REGISTRY_UNVERIFIED_INCIDENT_TYPE,
                tier,
                skip_if=_is_sticky_payload_mismatch,
            )
        return

    def _fields_for_existing(existing_row: "dict | None") -> "tuple[str, str, dict]":
        # Sticky-payload-status guard (see docstring): a limited-authority
        # RECORD must never overwrite an existing PAYLOAD_MISMATCH marker
        # with its own weaker status — record_incident replaces `detail`
        # wholesale, so without this the marker a full-authority caller
        # recorded would be silently lost the next time a restamp-only
        # caller records anything. Evaluated under the lock (see
        # _is_sticky_payload_mismatch) so the decision and the write are
        # atomic.
        recorded_status = unhealthy_status
        if not resolves_payload_status and _is_sticky_payload_mismatch(existing_row):
            recorded_status = PAYLOAD_MISMATCH

        severity = "failed" if _is_primary_adapter(tier) else "info"
        if recorded_status == KEY_COUNT_MISMATCH:
            summary = (
                f"Tier '{tier}' manifest key_count disagrees with its registry's "
                f"active-key count since the last verified store step"
            )
        elif recorded_status == KEYS_WITHOUT_SLOT:
            summary = (
                f"Tier '{tier}' registry holds active keys but no written payload "
                f"slot candidate exists since the last verified store step"
            )
        elif recorded_status == PAYLOAD_MISMATCH:
            summary = (
                f"Tier '{tier}' bound slot payload no longer matches its manifest "
                f"digest since the last verified store step"
            )
        else:
            summary = (
                f"Tier '{tier}' registry could not be verified against its slot "
                f"manifests ({recorded_status}) since the last verified store step"
            )
        return (
            severity,
            summary,
            {
                "tier": tier,
                "status": recorded_status,
                "detail": detail,
                "candidate_count": candidate_count,
                "action_hint": (
                    "restore this tier from a snapshot bundle via POST /backup/restore "
                    "and restart; see GET /integrity"
                ),
            },
        )

    # Literal severity/summary/detail below are placeholders overridden by
    # fields_for_existing on every call — computed under the lock, from the
    # row as it stands at write time, per the sticky-payload-status guard.
    record_incident(
        state_dir,
        type=_TIER_REGISTRY_UNVERIFIED_INCIDENT_TYPE,
        key=tier,
        severity="failed" if _is_primary_adapter(tier) else "info",
        summary="",
        detail={},
        fields_for_existing=_fields_for_existing,
    )


def _record_unverified_tier_incidents(config, tier_bindings: dict) -> None:
    """Record (or clear) a ``tier_registry_unverified`` incident per tier.

    Post-fold drift detection ONLY — called from the two event-kind fold
    finalizers (:func:`_finalize_interim`, :func:`_finalize_full`) right
    after :func:`_revalidate_adapter_manifests`, so a tier whose on-disk
    registry↔slot binding breaks sometime AFTER the last successful boot/lift
    store step (:func:`_hydrate_memory_store_in_place`) is still observed and
    reported before the next boot or lift re-runs that step. This is
    reporting only — nothing here changes what the live store currently
    serves for the tier; the store-publish boundary itself no longer has a
    per-tier unpublishable arm to report on (a tier that fails verification
    THERE quarantines the whole store instead — see
    :func:`_enter_store_quarantine` — a single ``store_quarantined``
    incident, not a per-tier one).

    Thin per-event driver over :func:`_record_or_resolve_tier_health` — the
    extracted body now also serves :func:`_stale_mark_keys` (the erase
    door), so this function contributes only the fold-specific input
    shape: one :class:`~paramem.adapters.registry_binding.TierBinding` per
    tier.

    Narrow, per-event semantics only: this call only ever records or
    resolves incidents for the tier(s) named in *tier_bindings*. A tier
    that goes unmentioned here — because it sat untouched by this event —
    keeps whatever incident state it already has; that incident persists
    until the SAME tier appears in a later call's *tier_bindings* (that
    tier's own next publish) and resolves it. Both current callers pass a
    fold's own publish verdict
    (:meth:`~paramem.training.consolidation.ConsolidationLoop.run_build_and_publish`'s
    ``result["tier_bindings"]``) — the tier(s) that ONE event's ledger
    names, never the whole tree.

    Args:
        config: Live server config object; only ``paths.data`` is read.
        tier_bindings: ``{tier: TierBinding}`` — a fold finalizer's own
            publish verdict, scoped to this event's tier(s) only.
    """
    for tier, binding in tier_bindings.items():
        _record_or_resolve_tier_health(
            config,
            tier=tier,
            unhealthy_status=None if binding.publishable else binding.status,
            detail=binding.detail,
            candidate_count=binding.candidate_count,
        )


_STORE_QUARANTINE_INCIDENT_TYPE = "store_quarantined"
_STORE_QUARANTINE_INCIDENT_KEY = "store"


def _enter_store_quarantine(
    config, exc: BaseException | None = None, *, reason: str | None = None
) -> None:
    """Quarantine the memory store: set the marker, record the ONE incident.

    ONE marker shape (``_state["store_quarantine"]`` = ``{"cause", "quarantined_at"}``)
    and ONE incident type (:data:`_STORE_QUARANTINE_INCIDENT_TYPE`) for BOTH
    of quarantine's two entries — the cause distinguishes which entry
    produced it, never a second marker or incident type:

    - The INVOLUNTARY entry — called from :func:`_hydrate_memory_store_in_place`
      when the store step's build/verify raises. Pass *exc*; the cause is
      the exception's type and message. Both
      :class:`~paramem.memory.store.BookkeepingInvariantViolation` and
      :class:`~paramem.adapters.registry_binding.TierBindingUnpublishable`
      already format their message with the offending tier and key(s), so
      no separate structured tier/key field is needed here — the message
      carries it.
    - The DELIBERATE entry — a repair door about to rewrite the tier tree
      (``POST /backup/restore``, the base-swap branch of
      ``POST /migration/rollback``) that has no exception to report because
      nothing has failed yet; the mutation itself is the reason the store
      must go offline first. Pass *reason*, a short human description of
      the action in progress (e.g. ``"restoring backup 20260421-04000012"``).
      The cause is synthesised with the SAME two keys an involuntary cause
      carries (``exception_type="StoreOfflineForRepair"``, ``message=reason``)
      so every cause reader — :func:`refusal_for`'s quarantine formatting,
      the incident detail — needs no branch for which entry produced it.

    Exactly one of *exc* / *reason* is given.

    No on-disk artifact beyond the incident row — the marker is process
    state (``_state["store_quarantine"]``) that a restart re-derives by
    re-running the same store step, never a durable file of its own.

    Args:
        config: Live server config object; only ``paths.data`` is read.
        exc: The exception caught at the store step (involuntary entry).
        reason: Human-readable description of the deliberate action about
            to rewrite the tree (deliberate entry).

    Raises:
        ValueError: Neither *exc* nor *reason* was given.
    """
    if exc is not None:
        cause = {"exception_type": type(exc).__name__, "message": str(exc)}
    elif reason is not None:
        cause = {"exception_type": "StoreOfflineForRepair", "message": reason}
    else:
        raise ValueError("_enter_store_quarantine requires exc or reason")

    _state["store_quarantine"] = {
        "cause": cause,
        "quarantined_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    logger.error(
        "Memory store quarantined (%s: %s) — the parametric-memory serving "
        "arm is out; every other capability proceeds normally (model, "
        "STT/TTS, HA entity graph, tri-path routing).",
        cause["exception_type"],
        cause["message"],
    )
    record_incident(
        data_state_dir(config.paths.data),
        type=_STORE_QUARANTINE_INCIDENT_TYPE,
        key=_STORE_QUARANTINE_INCIDENT_KEY,
        severity="failed",
        summary=f"Memory store quarantined: {cause['exception_type']}: {cause['message'][:160]}",
        detail={
            **cause,
            "action_hint": (
                "restore the affected tier from a snapshot bundle via "
                "POST /backup/restore; see GET /integrity"
            ),
        },
    )


def _clear_store_quarantine(config) -> None:
    """Resolve the store-quarantine incident and clear the marker.

    Called only from :func:`_hydrate_memory_store_in_place` on a successful
    store step — the record-and-clear-same-site pattern
    :func:`_record_unverified_tier_incidents` also uses. Idempotent: a
    store step that was never quarantined finds nothing to resolve.

    Args:
        config: Live server config object; only ``paths.data`` is read.
    """
    _state["store_quarantine"] = None
    resolve_incident(
        data_state_dir(config.paths.data),
        _STORE_QUARANTINE_INCIDENT_TYPE,
        _STORE_QUARANTINE_INCIDENT_KEY,
    )


def _hydrate_memory_store_in_place(store, config, *, model, tokenizer) -> bool:
    """The boot store step, and the lift: build, verify, and publish — or quarantine.

    Delegates the rebuild to :func:`_build_store_contents`. On success,
    publishes the three new structures via
    :meth:`~paramem.memory.store.MemoryStore.swap` (no reader ever observes
    a torn state) and resolves any open store quarantine
    (:func:`_clear_store_quarantine`). On failure — ANY exception from the
    build, most commonly
    :class:`~paramem.adapters.registry_binding.TierBindingUnpublishable`
    (a tier's registry↔slot binding did not verify) or
    :class:`~paramem.memory.store.BookkeepingInvariantViolation` (a known
    key has no bookkeeping row), but not limited to that family — *store*
    is left completely UNTOUCHED (``store.swap`` never runs: no partial
    publish, ever) and the store is quarantined
    (:func:`_enter_store_quarantine`). A fatal CUDA context fault
    (:func:`~paramem.utils.vram_guard.is_fatal_cuda_fault`) is NOT caught
    here: :func:`_build_store_contents` already re-raises it unchanged from
    its own probe, and it propagates through this function to the lifespan
    fail-fast handler — a poisoned CUDA context is recovered by a process
    restart, never by quarantining the store.

    This ONE function is both:

    - **The boot store step** — called from :func:`_preload_memory_store`
      (boot, and every in-process config apply / base-model reload) after a
      fresh :class:`MemoryStore` has been constructed and the base-swap
      gate has passed.
    - **The lift** — the identical, re-runnable primitive the repair flows
      invoke (via :func:`_lift_quarantined_store` →
      :func:`_preload_memory_store`) to re-attempt hydration without a
      restart or model churn: the quarantined branch of
      ``POST /debug/erase-keys`` after its file surgery, and
      ``POST /backup/restore``'s same-base convergence after the bundle
      rewrites the tier tree. The base-swap branch of
      ``POST /migration/rollback`` reaches it indirectly — its full
      release+reload re-runs :func:`_preload_memory_store` as part of the
      component rebuild. It runs fine with NO MODEL RESIDENT: passing
      ``model=None, tokenizer=None`` defers entry preload exactly as a
      cloud-only boot already does inside :func:`_build_store_contents` —
      registry and bookkeeping hydration need no model. The entry cache
      stays as it was until a model is resident again; a key the cache
      lacks in the meantime answers nothing at serving (no on-miss
      fallback), until the next model-resident fill act re-warms it.

    The post-fold entry cache is refilled by ``run_build_and_publish``'s
    ``adopt_increments`` instead, inside the same locked act that takes the
    fold's bundle live — :func:`_finalize_full` never calls this function.

    **NO-BASE-MODEL-PINNING INVARIANT** — enforced inside
    :func:`_build_store_contents`.  The caller passes ``model`` and
    ``tokenizer`` as direct kwarg expressions (never via a caller local), and
    the builder sets its WeightMemorySource frame-local to ``None`` before
    return.

    Parameters
    ----------
    store:
        Live :class:`~paramem.memory.store.MemoryStore` to hydrate in place.
        Must be the shared singleton (``_state["memory_store"]`` /
        ``loop.store``) so all holders (router, consolidation loop, /debug/dump)
        observe the rebuild without re-wiring. Left untouched when this
        call quarantines.
    config:
        Live server config object.
    model:
        Base model handle passed directly as a kwarg expression — do NOT bind to
        a caller local before passing. ``None`` when no model is resident
        (the lift during a cloud-only deferral, or a cloud-only boot).
    tokenizer:
        Tokenizer handle — same constraint.

    Returns
    -------
    bool
        ``True`` when *store* was published (the common case). ``False``
        when this pass quarantined instead — *store* was left untouched and
        the caller must not treat it as live.
    """
    try:
        new_entries, new_registry, new_bookkeeping, stats = _build_store_contents(
            config,
            model=model,
            tokenizer=tokenizer,
        )
    except Exception as exc:  # noqa: BLE001 — the store step's quarantine boundary
        if is_fatal_cuda_fault(exc):
            raise
        _enter_store_quarantine(config, exc)
        return False

    store.swap(new_entries, new_registry, new_bookkeeping)

    _state["store_preload_complete"] = stats["preload_complete"]
    _clear_store_quarantine(config)
    return True


def _preload_memory_store(config, *, model, tokenizer):
    """Build the MemoryStore, load registries, and hydrate the active-key cache.

    Called by :func:`_build_runtime_components`.  Returned store is assigned
    to ``_state["memory_store"]`` by the caller — ``None`` when the store
    step quarantined, so the caller leaves ``_state["memory_store"]`` unset
    and every non-store boot step still proceeds.

    Source selection uses ``config.consolidation.mode`` (NOT
    ``_state["mode"]``).  This prevents conflating the consolidation
    persistence mode (train/simulate) with the runtime mode (local/cloud-only).

    ``_state["store_preload_complete"]`` lifecycle:
    - Set ``True`` when the fill act ran to whatever extent the venue could
      serve (a shortfall is telemetry — a ``preload_recall_incomplete``
      incident — never a completeness failure).
    - Set ``True`` when ``config.inference.preload_cache=False`` (intentional
      opt-out — nothing to fill) or when there are no active keys to fill.
    - Set ``False`` while a base-model swap is in flight (empty store, early
      return) — the on-disk registry describes the PREVIOUS model and is
      invalid for the loaded one (see the base-swap gate below) — and when the
      fill is deferred for lack of a resident model on the train venue
      (:func:`~paramem.memory.source.train_venue_deferred`).  Both leave the
      caller's re-probe gate armed to retry on the next act.
    - Left as whatever it already was when the store step quarantines — a
      quarantine is a distinct, more severe condition surfaced via
      ``_state["store_quarantine"]``, not folded into this flag.

    The infrastructure integrity check below still runs — with ``store=None``
    — even when the store step quarantines: it is a read-only diagnostic
    over the on-disk tree, not a re-read of the (unpublished) store's
    content, so ``_state["integrity_check_failed"]`` stays accurate
    regardless of whether the store itself quarantined.

    The ``WeightMemorySource`` is kept as a frame-local and dropped on return —
    mirrors the no-frame-retention pattern of ``_load_model_into_state`` so the
    base model is not pinned past the preload.  This invariant is enforced inside
    :func:`_hydrate_memory_store_in_place` which this function delegates to after
    construction and the base-swap gate.

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
    MemoryStore | None
        The fully-constructed store (registries loaded; entries hydrated when
        ``preload_cache=True`` and the source probe succeeded), or ``None``
        when the store step quarantined instead of publishing.
    """
    from paramem.memory.store import MemoryStore as _MemoryStore

    memory_store = _MemoryStore()

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

    _swap_marker = _read_trial_marker(data_state_dir(config.paths.data).resolve())
    if _swap_marker is not None and _swap_marker.migration_kind == "base_swap":
        _state["integrity_check_failed"] = False
        _state["store_preload_complete"] = False
        logger.info(
            "preload_cache: base-swap in flight (phase=%s) — on-disk registry "
            "describes the previous model; live store starts empty until Phase B "
            "retrains the new model.",
            _swap_marker.base_swap_phase or "?",
        )
        return memory_store

    # Delegate registry load + entry hydration to the shared helper so the same
    # path is used at boot and at post-consolidation re-hydration.  A quarantine
    # leaves memory_store untouched (still the fresh, empty instance above) —
    # it is discarded below (this function returns None) rather than treated
    # as live.  The integrity check still runs (with store=None): it is a
    # read-only diagnostic over on-disk files, not a tolerant re-read of the
    # store's content, so it stays the authoritative integrity_check_failed
    # signal (surfaced via GET /integrity and the active-store migration
    # gate) regardless of whether the store itself quarantined.
    _hydrated = _hydrate_memory_store_in_place(
        memory_store, config, model=model, tokenizer=tokenizer
    )

    # Infrastructure integrity check — runs after all loaders so a published
    # store is fully populated for cross-consistency checks (store=None when
    # this pass quarantined — verify_infrastructure_integrity tolerates that,
    # same as the cold /integrity GET path before any preload has run). A
    # corrupt registry blocks migrations and flags integrity_check_failed (a
    # corrupt registry is a different, more severe condition than a deferred
    # or partial cache fill).
    #
    # cleanup_partial_slots (scratch left by interrupted training) no longer
    # runs here — it moved pre-mount, into _sweep_keyless_tier_artifacts
    # (paramem/server/app.py), called from _mount_adapters_from_slots, which
    # runs strictly before this function on every boot/reload that loads a
    # local model.  Removing an incomplete slot AFTER _hydrate_memory_store_in_place
    # (a few lines above) had already computed this tier's binding for
    # publish would let the mount stage, the store-publish builder, and this
    # integrity check disagree about the same on-disk tree within one boot;
    # running it pre-mount instead means all three read the same
    # already-cleaned tree.  A cloud-only boot never reaches
    # _mount_adapters_from_slots (no local model to mount), so this pass
    # does not run for it — the next boot/reload that acquires a local model
    # runs it, pre-mount, before that boot's own bindings are computed.
    try:
        from paramem.backup import key_store as _key_store_mod
        from paramem.backup.integrity import verify_infrastructure_integrity

        _daily_ok_local = _key_store_mod.daily_identity_loadable(
            _key_store_mod.DAILY_KEY_PATH_DEFAULT
        )
        _integrity_report = verify_infrastructure_integrity(
            config,
            store=memory_store if _hydrated else None,
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
            _state["integrity_check_failed"] = True
            logger.error(
                "Boot-time integrity check found %d failure(s); "
                "active-store migration will be refused until this is resolved",
                len(_integrity_report.failures),
            )
        else:
            _state["integrity_check_failed"] = False
            logger.info(
                "Boot-time integrity check passed (%d checks)",
                len(_integrity_report.checks),
            )
    except Exception:
        logger.exception(
            "Boot-time integrity check raised unexpectedly; integrity_check_failed left unchanged"
        )

    return memory_store if _hydrated else None


def _lift_quarantined_store(config) -> bool:
    """Re-run the store step against the resident process, model-optional —
    the ONE re-runnable "lift" a repair flow invokes to bring the memory
    store back online without a restart or a model reload.

    Delegates to :func:`_preload_memory_store` — the same fresh-``MemoryStore``
    -construct-then-hydrate path a config apply or a base-model reload
    already runs — passing ``model=_state.get("model")`` (``None`` in
    cloud-only mode, which skips entry preload exactly as a cloud-only boot
    already does; registry and bookkeeping hydration need no model) so a
    resident local-mode server also gets its entry cache re-warmed from
    weights. On success, republishes ``_state["memory_store"]`` AND rebuilds
    ``_state["router"]`` against the new store object — the router's index
    was built by capturing the store instance at construction time, so
    reusing the pre-lift router's own ``.reload()`` would silently keep
    reading the discarded store; only a fresh :class:`QueryRouter` observes
    the swap. Quarantine clear/incident-resolve is NOT this function's job —
    it already happened inside :func:`_hydrate_memory_store_in_place` (called
    by :func:`_preload_memory_store`) as a side effect of the successful
    build.

    Callers: the quarantined branch of ``POST /debug/erase-keys`` (after its
    file surgery), the post-restore step of ``POST /backup/restore`` (after
    the bundle rewrites the tier tree), and the tail of
    :func:`_finish_resumed_event` (a pending event's resume that went fully
    live while the store was quarantined — the automatic heal for a
    crashed publish on a cold-born tier). Safe to call again after a failed
    lift — the quarantine marker and its incident are re-derived from the
    fresh attempt, not accumulated across retries.

    The resume caller is structurally different from the other two: it
    cannot null ``_state["consolidation_loop"]`` before calling this
    function, unlike the erase/restore doors (which null it, then let the
    next fold's lazily-cached loop recreate against the fresh store this
    function publishes) — the resume's ``loop`` local is already captured
    by the finalizer closure it is about to dispatch, so nulling the cached
    loop would leave that closure holding a discarded, empty-store object.
    The resume caller instead rebinds ``loop.store`` to the new
    ``_state["memory_store"]`` itself, in place, immediately after this
    function returns ``True``; this function's own contract (publish
    ``_state["memory_store"]`` + rebuild ``_state["router"]``) is unchanged
    either way.

    Args:
        config: Live server config object.

    Returns:
        ``True`` when the store was published (quarantine cleared, if it was
        set) and the router rebuilt; ``False`` when the store step
        quarantined instead — ``_state["memory_store"]`` is ``None`` and the
        cause is in ``_state["store_quarantine"]``.
    """
    new_store = _preload_memory_store(
        config, model=_state.get("model"), tokenizer=_state.get("tokenizer")
    )
    _state["memory_store"] = new_store
    if new_store is None:
        return False
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        memory_store=new_store,
        ha_graph=_state.get("ha_graph"),
        intent_config=config.intent,
    )
    return True


def _report_intent_classifier_health(config, encoder_handle, exemplar_bank) -> None:
    """Report the embeddings intent classifier's load outcome as an incident.

    ``mode="embeddings"`` has no LLM fallback (unlike ``mode="llm"``, which
    slides to the encoder residual when no classifier model is registered),
    so a missing encoder or exemplar bank means every query classifies as
    ``Intent.UNKNOWN`` from here on: no personal-memory access, and cloud
    escalation stays available.  That capability loss is silent unless an
    operator is watching logs, so it is recorded as a durable incident
    (``type="intent_classifier_unavailable"``) surfaced via ``GET /status``
    and the attention block.

    A clean load is the success this incident resolves on, and this is the
    only site that observes it: both handles present clears the record rather
    than leaving a repaired classifier flagged on ``GET /status`` forever.

    No-op when ``config.intent.enabled`` is false or ``config.intent.mode``
    is not ``"embeddings"`` — a disabled classifier is not a recovered one.
    """
    if not (config.intent.enabled and config.intent.mode == "embeddings"):
        return
    if encoder_handle is not None and exemplar_bank is not None:
        resolve_incidents_by_type(
            data_state_dir(config.paths.data), "intent_classifier_unavailable"
        )
        return
    missing = []
    if encoder_handle is None:
        missing.append("encoder")
    if exemplar_bank is None:
        missing.append("exemplars")
    record_incident(
        data_state_dir(config.paths.data),
        type="intent_classifier_unavailable",
        key=config.intent.mode,
        severity="warning",
        summary=(
            "Intent classifier unavailable (mode=embeddings); queries now "
            "route as UNKNOWN — no personal-memory access, escalation allowed"
        ),
        detail={"mode": config.intent.mode, "missing": missing},
    )


def _build_runtime_components(
    config,
    *,
    cloud_only: bool,
    rebuild_session_buffer: bool = True,
    full_rebuild: bool = True,
) -> None:
    """Construct every runtime component into ``_state`` from stable first-level configuration.

    Single idempotent routine called by BOTH the lifespan startup and the
    live-apply path.  Replaces the lifespan's inline construction blocks at
    ``app.py:1546-1732`` + ``1753-2017`` (excluding the build-once post-load
    VRAM gate at ``1733-1751`` — the build-once post-load gate runs inline in the lifespan).

    **EXCLUDES** strictly-once lifespan concerns (signal handlers, asyncio
    tasks, timer reconciliation, mode init) and the build-once post-load VRAM
    gate.  The gate runs inline in the lifespan AFTER this routine returns so
    the measured allocation includes the STT/TTS GPU footprint.
    The gate is incompatible with this routine (it ``sys.exit(1)``s and reads
    a lifespan-frame local ``base_pred``).  The apply path's VRAM safety is
    the ``mem_get_info``-based fit-check inside ``_live_reload_base_model``.

    **Wyoming listener sockets are NOT re-bound here.** The lifespan binds
    them once with provider lambdas so profile swaps re-point automatically.
    The apply path must not call ``start_wyoming_server`` /
    ``start_wyoming_tts_server`` again.

    Construction ordering:

    1. session_buffer (snapshot old → construct → rehydrate → load_snapshot)
       — only when ``rebuild_session_buffer=True`` AND ``full_rebuild=True``.
    2. speaker_store (+ embedding model)  — ``full_rebuild=True`` only.
    3. STT/TTS managers: construct fresh from ``config`` → flip voice_box
       — ``full_rebuild=True`` only.
    4. cloud_agent + cloud_providers  — ``full_rebuild=True`` only.
    5. ha_client (close old → construct new) + ha_graph
       — ``full_rebuild=True`` only.
    6. memory store preload (``_preload_memory_store``) → assign
       ``_state["memory_store"]`` — always (re-probe gate may skip probe on warm).
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
        When ``True`` (default, boot + apply path): rebuild ALL runtime
        components (steps 1–9 above).
        When ``False`` (plain ``/gpu/acquire`` + auto-reclaim same config):
        skip the expensive, potentially network-touching steps (speaker_store,
        STT/TTS construction, cloud_agent, ha_client reconnect, exemplar banks,
        language_tracker).  Only steps 6 (memory-store probe, re-probe-gated) and 7
        (Router re-point) are run, plus ``set_classifier_model`` to register
        the freshly reloaded model handle (step 8 partial).  This avoids
        spurious STT/TTS ``load()`` calls and HA ``health_check()`` /
        ``load_entity_map()`` / ``get_services()`` network calls on every
        plain warm reclaim (config did not change — skip network reconnect).
    """
    # ── 1. session_buffer ────────────────────────────────────────────────────
    # Only on full_rebuild paths (boot + apply); plain reclaim keeps the live
    # buffer to avoid losing in-flight state.
    if full_rebuild and rebuild_session_buffer:
        # Save snapshot on the OLD buffer FIRST (mirrors the shutdown-teardown snapshot)
        # so mid-turn state (_turns, _sessions, and _open routing state) round-trips
        # when an encryption key is loaded (ensures continuity across key-loaded
        # deploys). No-op on a no-key deployment.
        old_buffer = _state.get("session_buffer")
        if old_buffer is not None:
            old_buffer.save_snapshot()

        _state["session_buffer"] = SessionBuffer(
            config.session_dir,
            retain_sessions=config.consolidation.retain_sessions,
            debug=config.debug,
            idle_timeout_minutes=config.session.idle_timeout_minutes,
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

        # ── 3. STT / TTS managers (construct-then-flip) ─────────────────────
        # Construct fresh managers from config, then atomically flip voice_box.
        # DO NOT call start_wyoming_server / start_wyoming_tts_server here —
        # those are lifespan-only.  The Wyoming provider lambdas re-point
        # automatically once voice_box is updated.
        #
        # Note: _set_voice_pipeline_profile only lazy-constructs when
        # _state["stt_gpu"] is None; it does NOT reconstruct when the config
        # changes.  The apply must construct fresh instances from config B and
        # install them before flipping.
        #
        # Flush the allocator-pool before the GPU STT load so vram_measure reads
        # accurate free-memory (mirrors _set_voice_pipeline_profile which calls
        # safe_empty_cache before load).
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
                # Flush allocator-pool before GPU STT load so vram_measure
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
                    _state["vram_components"]["stt"] = stt_gpu.vram_delta_bytes
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
                with vram_measure("tts") as _tts_vm:
                    tts_gpu.load_all()
                if tts_gpu.is_loaded:
                    _state["tts_gpu"] = tts_gpu
                    _state["vram_components"]["tts"] = _tts_vm["delta"]
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

        # ── 4. Cloud agent + providers ─────────────────────────────────────────
        _state["cloud_agent"] = get_cloud_agent(
            config.cloud_agent, cloud_enabled=config.cloud.enabled
        )
        if _state["cloud_agent"]:
            logger.info(
                "cloud agent: %s (%s)",
                config.cloud_agent.provider,
                config.cloud_agent.model,
            )
        else:
            logger.info("cloud agent: not configured")

        _state["cloud_providers"] = {}
        for name, provider_config in config.cloud_providers.items():
            agent = get_cloud_agent(provider_config, cloud_enabled=config.cloud.enabled)
            if agent:
                _state["cloud_providers"][name] = agent
                logger.info("cloud provider registered: %s (%s)", name, provider_config.model)
        logger.info("cloud providers available: %s", list(_state["cloud_providers"].keys()))

        # ── 5. HA client + ha_graph ──────────────────────────────────────────
        # Mandatory teardown: HAClient holds an httpx.Client pool.
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
        # and ha_graph — config did not change, no network reconnect needed.
        ha_graph = _state.get("ha_graph")
        logger.info(
            "_build_runtime_components: plain reclaim — skipping STT/TTS/HA/cloud/exemplar "
            "rebuild (full_rebuild=False, same config)"
        )

    # ── 6. Memory store preload ───────────────────────────────────────────────
    # Re-probe gate: re-probe only when (a) the store step's completion
    # record is absent (deferred or never run), or the store object itself
    # is absent, or (b) called from the apply path (caller sets
    # _state["_apply_config_in_progress"] = True before calling this routine
    # and clears it after).  On a plain reclaim with a complete store, still
    # rebuild the Router cheaply but skip the probe — never invalidated by a
    # GPU release: a cloud-only deferral does not invalidate the mirror.
    _do_probe = (
        not _state.get("store_preload_complete", False)
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
        # Complete store survives — keep it, just rebuild the Router below.
        logger.info(
            "_build_runtime_components: store-preload step already complete — "
            "skipping re-probe (re-probe gate); rebuilding router only"
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
            exemplar_bank = (
                load_exemplars(config.intent, encoder_handle)
                if encoder_handle is not None
                else None
            )
            _report_intent_classifier_health(config, encoder_handle, exemplar_bank)

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


def _remount_adapters_from_disk(config) -> None:
    """Re-mount every tier adapter from disk onto the resident model, on demand.

    Runs the boot mount machinery (:func:`_mount_adapters_from_slots`)
    AGAIN against an already-running server whose weight slots just changed
    underneath it — the case ``POST /backup/restore`` (same-base restore)
    needs after it rewrites the tier tree, and ``_load_model_into_state``
    does NOT cover: that function only ever mounts onto a FRESH base model
    at boot or after a full release+reload, never onto a model that already
    has these exact tier names mounted.

    NO-OP in cloud-only mode (``_state["model"] is None``): slots on disk
    are the truth and the next model acquisition mounts them — the same
    contract a cloud-only boot already has (see
    :func:`_preload_memory_store`'s docstring).

    When a model IS resident, every currently mounted tier adapter (main
    and interim alike — whatever the PRE-restore tree had mounted,
    including any orphan interim family the restore's clean-slate sweep
    just removed from disk) is disposed first via
    :func:`~paramem.models.loader.detach_adapters` — the ``PEFT``
    switch-before-delete discipline :func:`~paramem.memory.interim_adapter.unload_interim_adapters`
    also relies on. A same-base restore can legitimately replace every
    mounted tier adapter, so this may (and, for a full bundle, does) empty
    ``model.peft_config`` entirely — :func:`detach_adapters` documents this
    exact "the caller is emptying peft_config deliberately and owns the
    restore" branch. Emptying leaves ``model.active_adapter`` stale, but the
    model's object identity never changes: :func:`~paramem.models.loader.ensure_resident_tiers`
    restores every configured tier onto the SAME ``PeftModel`` object
    (a detach immediately followed by the create that repairs it, with no
    model use in between) rather than the model being unwrapped and
    re-wrapped.

    Caller responsibility: serialize this against every other GPU user —
    call under ``gpu_lock`` (mirrors every other PEFT-mutating primitive in
    this module). This function does not acquire the lock itself.

    Args:
        config: Live server config object.
    """
    from paramem.models.loader import detach_adapters, ensure_resident_tiers

    model = _state.get("model")
    if model is None:
        logger.info("Adapter re-mount skipped — cloud-only mode (no resident model)")
        return

    tokenizer = _state.get("tokenizer")
    mounted = sorted(model.peft_config.keys())
    detach_adapters(model, mounted)
    ensure_resident_tiers(model, config.tier_config_map())

    # Manifest status describes the PRE-restore tree; every row is stale the
    # instant the tree is rewritten underneath it — same reset
    # _load_model_into_state performs on every fresh load.
    _state["adapter_manifest_status"] = {}
    _mount_adapters_from_slots(model, tokenizer, config, _state)
    if "episodic" in model.peft_config:
        switch_adapter(model, "episodic")
    logger.info("Adapter re-mount complete")


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

    Refuses (``ConfigStoreMismatch``) on a config that contradicts the
    tier store already on disk — see
    :func:`~paramem.server.config_store_validator.check_config_against_store`.
    """
    apply_process_cap(fraction=config.vram.process_cap_fraction)
    logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)

    check_config_against_store(config)
    # The check just passed -- any config_refused incident from a prior
    # crashed reload (or a now-fixed store) is stale. One record site
    # (the reload's ConfigStoreMismatch handler) and one resolve site
    # (here) cover boot and every reload alike, since both paths reach
    # this line only through this one function.
    resolve_incidents_by_type(data_state_dir(config.paths.data), "config_refused")

    _tier_configs = config.tier_config_map()
    with vram_measure("base") as _base_vm:
        model, tokenizer = load_base_model(config.model_config, _tier_configs)
    # Store the measured delta in the per-component VRAM ledger (bytes).
    # vram_measure stores an INT — no BASE-MODEL HOLDER created here.
    _state["vram_components"]["base"] = _base_vm["delta"]

    # Manifest caches are model-specific; re-init on every load.
    _state["adapter_manifest_status"] = {}
    _state["base_model_hash_cache"] = {}

    # Mount adapters from slots onto the already-resident (cold) tiers.
    _mount_adapters_from_slots(model, tokenizer, config, _state)

    # Restore the main episodic adapter as the active adapter.
    if "episodic" in model.peft_config:
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
    ``pending_rehydration`` in ``_dispatch_consolidation``) with its own 1.0
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
) -> Literal["insufficient_vram", "reload_failed", "apply_failed", "config_refused"] | None:
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
    (cycle_count, promoted_keys, simhash sets) is not always
    persisted per-cycle and is load-bearing for promotion + debug
    snapshots.

    Voice-pipeline drain/restore (invariant owned by this primitive):
    This function drains the voice pipeline to CPU before its VRAM gate
    (when ``voice_profile=="gpu"``) and restores it after a successful
    PARTIAL reload (``refresh_config_from_disk=False``).  The FULL-rebuild
    path (``refresh_config_from_disk=True``) restores voice via
    ``_build_runtime_components`` — adding a restore here for that branch
    would double-load.  Failure branches leave voice on CPU so the
    cloud-only server holds ~0 GiB.  A voice-restore failure on the PARTIAL
    path is a handled, non-fatal degradation — logged and re-drained to CPU,
    never allowed to escape and strand the (already-loaded) base model
    behind the transient ``"live_reload"`` cloud-only reason.

    The drain is idempotent for cloud-only callers (``voice_profile=="cpu"``
    → ``_set_voice_pipeline_profile`` early-returns on a matching profile).
    It also closes the double-voice leak in ``_build_runtime_components``
    full-rebuild: the rebuild overwrites ``_state["stt_gpu"]``/``["tts_gpu"]``
    without unloading the old GPU instances; the drain unloads and nulls them
    first so the rebuild starts from a clean slate.

    Caller responsibilities:
    - Every dispatching caller — ``_apply_config_live`` (via
      ``gpu_lock_sync()``), ``_auto_reclaim_loop``, plain ``/gpu/acquire``,
      and the base-swap orchestration's post-Phase-B reload (all three via
      ``async with gpu_lock()`` around the ``run_in_executor`` dispatch) —
      holds the GPU lock across this call and passes ``lock_held=True`` so
      the internal ``_set_voice_pipeline_profile`` calls do not re-acquire
      the non-reentrant lock (deadlock). ``lock_held=False`` (the default)
      is for direct, non-dispatched callers that do not hold the lock.
    - Must accept ~25-30 s of model-load latency. Mode is flipped to
      cloud-only for the duration so any concurrent /chat handler
      routes to cloud rather than crashing on a None model.

    Parameters
    ----------
    refresh_config_from_disk:
        When ``True`` (the config-apply path, called from
        ``_apply_config_live``):

        - Re-read ``_state["config"]`` from disk via ``load_server_config``
          **before** ``_release_base_model_in_process`` — so the new config is
          committed before the release (recoverable if the reload then fails).
        - After a successful ``_load_model_into_state``, call
          ``_build_runtime_components(config, cloud_only=False,
          full_rebuild=True)`` to rebuild ALL first-level-configuration
          components (memory store preload, router, exemplar banks, STT/TTS
          managers, ha_client/ha_graph, etc.).
        - Flip ``_state["mode"]="local"`` after a clean full rebuild.  A
          partial preload (``preload_recall_incomplete`` incident recorded) is
          NOT a failure — the missed keys simply answer nothing until the
          mirror re-warms at the next fill act, so the server stays local
          and the incident surfaces via the generic /status attention
          collector.
        - On rebuild failure: stay cloud-only, set ``cloud_only_reason`` to
          ``"apply_failed"``.

        When ``False`` (plain ``/gpu/acquire`` + auto-reclaim, same config):

        - Model reload + ``set_classifier_model`` (to register the new handle).
        - ``_build_runtime_components`` is called with ``full_rebuild=False``
          (rebuilds memory-store probe [re-probe-gated] + Router re-point only).
          STT/TTS, HA reconnect, exemplar banks, and language_tracker are
          skipped — same config, no delta.
    rebuild_session_buffer:
        Threaded from ``_apply_config_live``: ``True`` when
        ``retain_sessions`` or ``debug`` changed between config A and config B,
        so the ``SessionBuffer`` is rebuilt.  Always ``False`` on the plain
        reclaim path (``refresh_config_from_disk=False``) — the session config
        did not change.
    lock_held:
        When ``True``, the caller already holds the shared non-reentrant
        threading.Lock from ``paramem/server/gpu_lock.py`` (``gpu_lock_sync()``
        or ``async with gpu_lock()``).  The internal
        ``_set_voice_pipeline_profile`` calls forward this flag so they skip
        re-acquisition.  Every dispatching caller — ``_apply_config_live``,
        ``_auto_reclaim_loop``, plain ``/gpu/acquire``, and the base-swap
        orchestration's post-Phase-B reload — holds the lock and passes
        ``True``.  ``False`` (the default) is for direct, non-dispatched
        callers that do not hold the lock.

    Returns
    -------
    Literal["insufficient_vram", "reload_failed", "apply_failed", "config_refused"] or None
        ``None`` on success (mode is now ``"local"``).  On a handled
        failure, the same reason string just written to
        ``_state["cloud_only_reason"]`` — one of the closed vocabulary
        ``"insufficient_vram"`` (the VRAM preflight gate refused the load,
        OR the load itself raised ``VramExhausted``),
        ``"reload_failed"`` (the model load raised anything else, or the
        plain-reclaim component rebuild raised after a successful load),
        ``"apply_failed"`` (the full config-apply component rebuild
        raised after a successful load), or ``"config_refused"`` (the
        store changed between an earlier validation and this load, so the
        load itself raised ``ConfigStoreMismatch`` — a race safety net,
        recorded as an incident; see
        :func:`~paramem.server.config_store_validator.check_config_against_store`).
        An exception that escapes this function (not caught by any of the
        above) signals an UNHANDLED failure — the caller cannot assume the
        process is in a known clean state and should treat it differently
        from a returned reason string.  This return value is the
        control-flow channel; ``_state["cloud_only_reason"]`` keeps being
        written on every path exactly as before because ``/status`` reads
        it directly.

    Note on the synchronous maintenance guard:
    When ``refresh_config_from_disk=True`` the CALLER (``_apply_config_live``)
    sets ``_state["mode"]="cloud-only"`` + ``cloud_only_reason="live_reload"``
    on the event loop BEFORE dispatching this function via ``run_in_executor``,
    so the consolidation dispatcher's ``mode != "local"`` defer fires
    (:func:`_consolidation_dispatch_guards` → ``deferred_cloud_only``).
    This function is NOT responsible for setting that guard.
    """
    # When refreshing config from disk, load the new config BEFORE releasing
    # the model — so the new config is committed to _state even if the reload
    # then fails (partial-rebuild recovery: _state["config"] is
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
    # Also closes the double-voice leak in _build_runtime_components
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
    # (``key_metadata.json``, ``indexed_key_registry.json``). The only state
    # that resets is
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
        return "insufficient_vram"

    failure_reason: Literal["insufficient_vram", "reload_failed", "config_refused"] | None = None
    refusal_message: str | None = None
    refusal_check: str | None = None
    try:
        _load_model_into_state(config)
    except VramExhausted:
        # Log here (the traceback is still live), but do NOT free here:
        # the partially-loaded model is pinned by this active traceback,
        # so ``safe_empty_cache`` would not return its bytes. The cleanup
        # runs below, after the except block drops the traceback. Same
        # terminal the boot path already maps VramExhausted to
        # (app.py's lifespan VramExhausted handler).
        logger.exception("Live model reload failed during base-model load — VRAM exhausted")
        failure_reason = "insufficient_vram"
    except ConfigStoreMismatch as exc:
        # The residual race this primitive's own refusal is the safety net
        # for: a door validated a candidate against the store, and the
        # store changed (a fold, POST /interim/discard, POST
        # /speaker/forget) before this reload re-checked it inside
        # _load_model_into_state. Log here (traceback still live; see the
        # note above) — the incident record and the release both run below,
        # after the traceback drops.  Capture the exception's STRING content
        # (message, check) rather than the exception object itself: holding
        # the object would pin its traceback across the release below,
        # falsifying this comment and defeating the release's own device-
        # memory reclaim (see the note on the ``except Exception`` branch).
        logger.exception("Live model reload refused — config contradicts the store")
        refusal_message = str(exc)
        refusal_check = exc.check
        failure_reason = "config_refused"
    except Exception:
        logger.exception("Live model reload failed during base-model load")
        failure_reason = "reload_failed"

    if failure_reason is not None:
        # (b) Fail clean. The traceback is gone now, so the partial model
        # is unreferenced — ``_release_base_model_in_process`` ->
        # ``safe_empty_cache`` (gc.collect + clearCublasWorkspaces +
        # empty_cache) actually returns its device memory, leaving the
        # cloud-only server at ~0 GiB. STT/TTS GPU teardown is the
        # caller's job: it must run outside the ``gpu_lock`` this function
        # may be holding (the auto-reclaim path calls us under that lock).
        _release_base_model_in_process()
        _state["cloud_only_reason"] = failure_reason
        if failure_reason == "config_refused":
            record_incident(
                data_state_dir(config.paths.data),
                type="config_refused",
                key=refusal_check,
                severity="failed",
                summary=f"Config refused on reload: {refusal_message.splitlines()[0][:160]}",
                detail={
                    "message": refusal_message,
                    "adapter_dir": str(config.adapter_dir),
                },
            )
            logger.error(
                "Live model reload refused — config contradicts the store on disk; "
                "server stays cloud-only, no restart. Fix the config or the store, "
                "then retry `pstatus --acquire`."
            )
        else:
            logger.error(
                "Live model reload failed — released partial allocation, "
                "server stays cloud-only until the GPU frees or a restart."
            )
        return failure_reason

    if refresh_config_from_disk:
        # Config-apply path: full component rebuild via the shared routine.
        # Signal the re-probe gate that this is an apply (probe even when store is warm).
        _state["_apply_config_in_progress"] = True
        rebuild_failed = False
        try:
            # rebuild_session_buffer is threaded from _apply_config_live
            # (True when retain_sessions or debug changed between A and B).
            _build_runtime_components(
                config,
                cloud_only=False,
                rebuild_session_buffer=rebuild_session_buffer,
                full_rebuild=True,
            )
        except Exception:
            logger.exception("Live config apply: _build_runtime_components failed")
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
            return "apply_failed"

        # Full rebuild succeeded.  A partial preload (a preload_recall_incomplete
        # incident recorded by _preload_memory_store inside
        # _build_runtime_components) is NOT a failure: the missed keys simply
        # answer nothing at serving until a later fill act (the next apply or
        # /gpu/acquire) re-warms the mirror, so the server stays local.  The
        # incident stays active as a signal — surfaced via the generic
        # /status attention collector — and is resolved when a later preload
        # fully hydrates.  The set_classifier_model call is inside
        # _build_runtime_components (step 8 exemplar banks).
        _state["mode"] = "local"
        _state["cloud_only_reason"] = None
        logger.info("Live config apply — complete; mode=local")
        result = None
    else:
        # Plain reclaim path (same config): rebuild Router + classifier handle.
        # Re-probe gate: _build_runtime_components skips the expensive weight-probe
        # when the store-preload step's completion record is present and
        # memory_store is non-None.  _apply_config_in_progress is NOT set here.
        rebuild_failed = False
        try:
            # full_rebuild=False — plain reclaim rebuilds only the memory
            # store (re-probe-gated), Router (re-point at warm store), and
            # set_classifier_model (re-register the new model handle).
            # STT/TTS construction, HA reconnect, cloud_agent, exemplar banks,
            # and language_tracker are skipped (same config, no delta).
            _build_runtime_components(
                config,
                cloud_only=False,
                rebuild_session_buffer=False,
                full_rebuild=False,
            )
        except Exception:
            logger.exception(
                "Live model reload: _build_runtime_components failed; "
                "intent/router may be stale until next reload or restart"
            )
            rebuild_failed = True

        if not rebuild_failed:
            # Partial-path success restore: the entry-drain moved voice to CPU;
            # put it back now that the base model is live.  The full-rebuild path
            # (refresh_config_from_disk=True) skips this — _build_runtime_components
            # already reconstructed voice on GPU and set voice_profile="gpu".
            #
            # A voice-restore failure is a handled, non-fatal degradation: this
            # primitive's responsibility is the base-model reload, and a local
            # server with voice stuck on CPU beats reporting cloud-only over a
            # base model that is actually resident (the restore failure may
            # itself be VRAM pressure from the voice models).  No cleanup
            # re-drain is needed: voice_profile is already "cpu" here (the
            # entry drain above is the only prior mutation, and a failed
            # "gpu" call never reaches the assignment that would flip it), so
            # a second _set_voice_pipeline_profile("cpu") call would be a
            # guaranteed no-op via its own idempotent profile-match guard.
            # Nor can the failure leave a GPU allocation behind: the
            # WhisperSTT/TTSManager constructors are pure attribute
            # assignment (no CUDA allocation), and the actual .load() calls
            # are internally try/excepted (best-effort) and cannot raise
            # past this point.
            try:
                _set_voice_pipeline_profile("gpu", lock_held=lock_held)
            except Exception:
                logger.warning(
                    "Live model reload: voice GPU restore failed after a successful "
                    "base-model reload — continuing in local mode with voice on CPU; "
                    "the next consolidation cycle's post-cycle voice restore or a "
                    "config apply retries the GPU voice restore.",
                    exc_info=True,
                )
            _state["mode"] = "local"
            _state["cloud_only_reason"] = None
            logger.info("Live model reload — complete; mode=local")
            result = None
        else:
            _release_base_model_in_process()
            _state["cloud_only_reason"] = "reload_failed"
            logger.error(
                "Live model reload: component rebuild failed after successful model load — "
                "released allocation, staying cloud-only."
            )
            result = "reload_failed"

    return result


# ════════════════════════════════════════════════════════════════════════════
#  INVARIANT — BASE-MODEL HOLDER REGISTRY  (cloud-only VRAM-leak guard)
#  Every reference to the base model (``_state["model"]``) must be dropped on
#  release so a cloud-only server holds ~0 GiB. Holders accumulate as new
#  components capture the model; a teardown that silently goes stale leaks the
#  whole base model (~4 GiB) — exactly what happened pre-2026-05-21.
#    • Find every holder:   grep -rn "BASE-MODEL HOLDER" paramem/
#    • Object / module-global holders  → drop them in this function.
#    • Lifespan-frame locals           → drop them IN THE LIFESPAN. This
#      function is called from TWO contexts: (a) externally — ``/gpu/release``
#      and ``_live_reload_base_model``, which run OUTSIDE the suspended
#      ``@asynccontextmanager`` frame and CANNOT reach frame locals; (b) from
#      inside the lifespan teardown (post-``yield`` block), where the frame IS
#      resumed.  In BOTH cases no lifespan-frame locals hold the model
#      (``_load_model_into_state`` keeps the model out of its caller's frame by
#      design; the only lifespan locals that held the model — ``_source`` and
#      ``_classifier_model`` — are nulled in their own subroutine frames before
#      ``yield``).  The holder-registry audit is the same for both call sites.
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

    NOTE: ``paramem.training.graph_tier.GraphTierRefiner`` also carries a
    ``# BASE-MODEL HOLDER`` tag (per the CLAUDE.md grep-registry
    convention) but is NOT a sixth holder above —
    ``ConsolidationLoop.build_tier_refiner`` constructs a fresh instance as a
    local variable on every call and drops it (no reference retained
    anywhere) when the caller returns, so it needs no release reach from
    this function.

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
    # Clear the base-model entry from the per-component VRAM ledger so
    # /status does not report stale VRAM after a cloud-only transition.
    _state.get("vram_components", {}).pop("base", None)
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


def _apply_config_live(*, force: bool = False) -> dict:
    """Apply the on-disk ``configs/server.yaml`` to the running server in-process.

    Acquires ``gpu_lock_sync`` with a bounded timeout, then:

    1. Re-checks ``_state["consolidating"]`` under the lock (guards against a
       pre-TRIAL cycle still running at accept/rollback time).
    2. Performs a no-op skip when the on-disk config hash equals
       ``_state["config_drift"]["loaded_hash"]`` (the hash of the config that
       was active in memory when the server last booted or accepted a migration).
       This is the rollback case — disk is back to A, memory is A.  Returns
       immediately without GPU churn (disk hash equals memory hash — config A is already active).
       Skipped entirely when *force* is ``True`` — see the *force* parameter
       below; the reload always runs on that path regardless of what the
       disk/memory hash comparison would have concluded.

       **Caller ordering precondition:** the accept handler MUST dispatch
       ``_apply_config_live`` BEFORE refreshing ``config_drift.loaded_hash``
       to config B.  If the refresh happens first, ``disk_hash == loaded_hash``
       fires on the accept path and the apply is incorrectly skipped.
       ``ServerConfig`` has no ``source_path`` attribute — the prior
       implementation that computed ``mem_hash`` via that attribute was always
       falling back to the live path, causing ``disk_hash == mem_hash`` on every
       call.
    3. Detects R-PORT / R-PATHS carve deltas:

       - ``stt.port`` / ``tts.port`` change → R-PORT carve:
         ``restart_required_reason in {"stt_port_change", "tts_port_change"}``.
         Performs a transient ``socket.bind`` pre-flight on the new port(s); on
         bind failure returns ``restart_eligible=False`` + a "port in use"
         reason.  On bind success returns ``restart_eligible=True`` so the
         CLI can prompt the operator and, on consent, run a fixed
         ``systemctl --user restart paramem-server`` via the
         ``paramem.utils.systemctl`` transport seam (``restart_hint`` is
         display-only text).  ``_apply_config_live`` itself never calls
         ``_restart_service``.
       - ``paths.sessions`` / ``paths.data`` change → R-PATHS carve:
         short-circuits BEFORE any live reload.
         ``restart_required_reason="paths_change"``, ``restart_eligible=False``.
         Data is NOT migrated automatically; operator must move adapters,
         registry, and sessions to the new root before restarting.
       - Mixed deltas (carve + non-carve fields): apply non-carve fields live
         first, then signal the carve.  A ``paths.*`` mix is always
         manual-restart regardless of other fields.

    3b. Reconciles both systemd timers — consolidation against config B's
        ``consolidation.refresh_cadence`` and scheduled backup against
        config B's ``security.backups.schedule`` — via
        :func:`_reconcile_scheduling_timers`, unconditionally and
        independent of the R-PORT/R-PATHS carve outcome. This is stateless:
        it re-reads the PRESENT values and acts on them — it never diffs
        against config A's former values, so a cadence-only or
        backup-schedule-only edit applies live (drift clears) without a
        restart. Skipped only when config B failed to load. Never raises —
        each timer's reconcile is independently logged and swallowed inside
        :func:`_reconcile_scheduling_timers`.
    4. Calls ``_live_reload_base_model(refresh_config_from_disk=True)`` for
       non-carve fields.
    5. On ``mode==local`` after the rebuild, calls
       ``_set_voice_pipeline_profile("gpu")`` (no-op if already gpu).

    **Caller contract (synchronous maintenance guard):**
    The CALLER must set the synchronous maintenance guard on the event loop
    BEFORE dispatching this function via ``run_in_executor``:

    .. code-block:: python

        _state["mode"] = "cloud-only"
        _state["cloud_only_reason"] = "live_reload"
        await loop.run_in_executor(None, _apply_config_live)

    This ensures the consolidation dispatcher's ``mode != "local"`` defer
    (:func:`_consolidation_dispatch_guards` → ``deferred_cloud_only``) fires
    before the executor runs.  This function does NOT set the guard
    internally (it runs in an executor thread, not on the event loop).

    Note: ``_live_reload_base_model`` also sets ``mode="cloud-only"`` directly
    as part of the drain sequence; the caller-set guard above is still required
    for the event-loop / scheduler visibility window before the executor thread
    even starts.

    **Lock contract:**
    This function acquires ``gpu_lock_sync`` internally (bounded timeout).
    ``_live_reload_base_model`` must NOT acquire it again — double-acquire on
    the non-reentrant ``threading.Lock`` deadlocks.

    Parameters
    ----------
    force:
        When ``True``, step 2's no-op skip is bypassed entirely — the reload
        always runs. The sole caller is the base-swap branch of
        ``POST /migration/rollback``: that branch has already entered store
        quarantine and rewritten the tier tree from a bundle before calling
        here, so a skipped reload would strand the quarantine with the base
        model never reloaded and the store never lifted. The skip logic
        itself stays intact for every other caller (migration accept, both
        ``POST /backup/restore`` branches, the non-base-swap rollback
        branch, and any future caller) — this parameter only lets ONE
        caller step around it, rather than the skip being weakened for
        everyone. ``False`` (default) preserves the existing
        disk-hash-vs-memory-hash comparison for every other caller.

    Returns
    -------
    dict
        ``{
            "applied_live": bool,
            "cloud_only_reason": str | None,
            "restart_required_reason": str | None,
            "restart_eligible": bool,
            "skipped": str | None,
        }``

        ``restart_eligible`` is ``True`` when an R-PORT carve pre-flighted
        successfully and the CLI may trigger a prompted restart via
        ``restart_hint``.  The server never fires the restart itself.
    """
    import socket as _socket

    from paramem.server.drift import compute_config_hash
    from paramem.server.gpu_lock import gpu_lock_sync

    # ── acquire GPU lock with bounded timeout ────────────────────────────────
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
            # No reload was attempted on this path, so there is no reload
            # outcome to report — the transient "live_reload" cloud-only
            # sentinel the caller pre-set is meaningless to the operator
            # (see restart_required_reason for the actual, named cause).
            "cloud_only_reason": None,
            "restart_required_reason": "lock_timeout",
            "restart_eligible": False,
            "skipped": None,
        }

    try:
        # ── re-check consolidating under the lock (TOCTOU guard) ────────────────
        if _state.get("consolidating", False):
            logger.warning(
                "_apply_config_live: a consolidation cycle is still running — "
                "apply aborted; config is on disk, restart or retry to apply"
            )
            return {
                "applied_live": False,
                # No reload was attempted on this path, so there is no reload
                # outcome to report — the transient "live_reload" cloud-only
                # sentinel the caller pre-set is meaningless to the operator
                # (see restart_required_reason for the actual, named cause).
                "cloud_only_reason": None,
                "restart_required_reason": "consolidating",
                "restart_eligible": False,
                "skipped": None,
            }

        # ── no-op skip (disk hash == memory hash → rollback already applied config) ──
        # Bypassed entirely when force=True — see the *force* parameter doc.
        config_path_str = _state.get("config_path")
        config_a = _state.get("config")
        live_config_path = Path(config_path_str) if config_path_str else DEFAULT_SERVER_CONFIG_PATH
        if not force and live_config_path.exists():
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
            # Caller ordering (document, do not implement here): the accept
            # handler MUST read and capture ``loaded_hash`` BEFORE refreshing it
            # to config B, then dispatch ``_apply_config_live`` with the
            # pre-refresh hash still in ``_state["config_drift"]["loaded_hash"]``.
            # If the accept handler refreshes ``loaded_hash`` BEFORE dispatching
            # the apply, disk_hash == loaded_hash would fire on the accept path
            # and the no-op skip would wrongly suppress the rebuild.
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
                    "restart_eligible": False,
                    "skipped": "no_change",
                }

        # ── config-A-vs-B carve classification (carve vs live-apply delta detection) ──
        # Load config B from disk without committing it yet.
        config_b = None
        if live_config_path.exists():
            try:
                config_b = load_server_config(live_config_path)
            except Exception:
                logger.exception(
                    "_apply_config_live: failed to load config B from disk for carve diff"
                )

        # ── scheduler reconcile: stateless re-read of the PRESENT cadence ──
        # Both systemd timers join the live-apply mechanism like every other
        # concern below: _reconcile_scheduling_timers re-reads
        # consolidation.refresh_cadence AND security.backups.schedule
        # straight from config B (the on-disk config being applied) and
        # reconciles each timer to it. Neither diffs against config A's
        # former values — a cadence-only or backup-schedule-only edit must
        # apply live without a restart, so drift can clear on its own
        # instead of only clearing via a restart that would then re-arm the
        # very timer that fired the alert (self-prophecy). This runs even
        # when the reload below carves off into a manual-restart path
        # (R-PATHS / R-PORT) — the schedules are independent concerns from
        # the base-model reload, and the operator still gets live-applied
        # cadences while the carve is pending a restart. TRIAL/migration
        # semantics are preserved for free: this function (and therefore
        # this reconcile) is only reached from accept/rollback, never from
        # migration_confirm's disk-only candidate write, so a TRIAL
        # candidate's schedules never arm the live timers.
        if config_b is not None:
            _reconcile_scheduling_timers(config_b)

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
                    # No reload was attempted on this path, so there is no reload
                    # outcome to report — the transient "live_reload" cloud-only
                    # sentinel the caller pre-set is meaningless to the operator
                    # (see restart_required_reason for the actual, named cause).
                    "cloud_only_reason": None,
                    "restart_required_reason": "paths_change",
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
                        # No reload was attempted on this path, so there is no reload
                        # outcome to report — the transient "live_reload" cloud-only
                        # sentinel the caller pre-set is meaningless to the operator
                        # (see restart_required_reason for the actual, named cause).
                        "cloud_only_reason": None,
                        "restart_required_reason": carve_reason,
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
                        # No reload was attempted on this path, so there is no reload
                        # outcome to report — the transient "live_reload" cloud-only
                        # sentinel the caller pre-set is meaningless to the operator
                        # (see restart_required_reason for the actual, named cause).
                        "cloud_only_reason": None,
                        "restart_required_reason": restart_required_reason,
                        "restart_eligible": True,
                        "skipped": None,
                    }
                # Mixed delta: fall through to reload; carve signalled in return dict.

        # ── compute retain_sessions / debug delta → rebuild_session_buffer ─
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
        reason = _live_reload_base_model(
            refresh_config_from_disk=True,
            rebuild_session_buffer=_rebuild_session_buf,
            lock_held=True,
        )
        applied_live = reason is None

        if applied_live:
            # Voice pipeline was set by _build_runtime_components inside
            # _live_reload_base_model; final no-op profile flip to confirm gpu.
            _set_voice_pipeline_profile("gpu", lock_held=True)

        return {
            "applied_live": applied_live,
            "cloud_only_reason": reason,
            "restart_required_reason": restart_required_reason,
            "restart_eligible": restart_eligible if applied_live else False,
            "skipped": None,
        }

    finally:
        lock_ctx.__exit__(None, None, None)


async def _apply_config_live_guarded(*, force: bool = False) -> dict:
    """Dispatch ``_apply_config_live`` under the synchronous maintenance guard.

    Sole owner of the guard+dispatch+restore pattern shared by the migration
    accept, base-swap-rollback, and migration_rollback handlers.  Sets the
    cloud-only guard (``mode="cloud-only"``,
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

    Args:
        force: Forwarded to :func:`_apply_config_live` — see its own *force*
            parameter doc. Only the base-swap branch of
            ``POST /migration/rollback`` passes ``True``.

    Returns the ``_apply_config_live`` result dict (``applied_live``,
    ``restart_required_reason``, ``restart_eligible``, ...).  Callers that only
    need the mode-restore side effect may ignore the return.
    """
    _prior_mode = _state.get("mode")
    _prior_cloud_only_reason = _state.get("cloud_only_reason")
    _state["mode"] = "cloud-only"
    _state["cloud_only_reason"] = "live_reload"

    loop = asyncio.get_running_loop()
    # Call with zero args in the (overwhelmingly common) force=False case —
    # every existing test double patching _apply_config_live at module scope
    # was written against that zero-arg signature; only a caller that
    # explicitly opts into force=True (see _apply_config_live's own
    # docstring) needs the kwarg-carrying partial.
    call = functools.partial(_apply_config_live, force=True) if force else _apply_config_live
    apply_result = await loop.run_in_executor(None, call)

    if _state.get("mode") == "cloud-only" and _state.get("cloud_only_reason") == "live_reload":
        _state["mode"] = _prior_mode
        _state["cloud_only_reason"] = _prior_cloud_only_reason
    return apply_result


async def _gpu_release_internal():
    """Unload the base model in-process and switch to cloud-only mode.

    Body of the ``/gpu/release`` handler, extracted so
    ``_run_base_swap_orchestration``'s fresh-start path can invoke the same
    lock-guarded teardown directly — the route handler's ``base_swap_active``
    guard exists to refuse *external* callers during an active swap and would
    wrongly 409 on the orchestration's own in-flight flag.

    Starting the auto-reclaim loop is deliberately NOT done here — it is
    route policy ("step aside for an external consumer, come back later"),
    owned by the ``/gpu/release`` handler. The base-swap orchestration's
    fresh-start reload calls this function only to release ahead of its own
    immediate ``_apply_config_live`` reload; a reclaim task started here
    could fire and reload the OLD base after a deferred swap, flipping
    ``mode`` back to ``"local"`` so ``/gpu/acquire``'s deferred-Phase-B
    relaunch (gated on ``mode == "cloud-only"``) would never trigger.

    Runs under ``gpu_lock`` (mirroring ``/gpu/acquire``'s reload dispatch):
    the synchronous teardown (``_release_base_model_in_process``) executes
    off the event-loop thread via ``run_in_executor``, and the consolidating
    flag is re-checked *inside* the lock — the async lock acquire is itself
    an await point, so a cycle could start between the caller's pre-lock
    check and lock acquisition without this recheck (mirrors
    ``_apply_config_live``'s in-lock TOCTOU guard). The mode flip to
    ``cloud-only`` happens inside the lock too, closing the window where a
    chat leg could win the lock with ``mode`` still ``"local"`` and no
    resident model. The voice-profile swap runs OUTSIDE the lock — it
    acquires ``gpu_lock_sync`` itself, and nesting would deadlock the
    non-reentrant lock (matches ``/gpu/acquire``'s deliberate pattern).

    Returns:
        dict ``{"mode": "cloud-only", "released": True, "reason": "released"}``
            on success.
        ``JSONResponse`` 503 ``{"error": "consolidating", ...}`` if a
            consolidation cycle is found in flight under the lock.
    """
    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
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

        logger.info(
            "Release requested via /gpu/release — unloading model and switching to cloud-only."
        )

        await asyncio.get_running_loop().run_in_executor(None, _release_base_model_in_process)

        _state["mode"] = "cloud-only"
        _state["cloud_only_reason"] = "released"

    await asyncio.get_running_loop().run_in_executor(None, _set_voice_pipeline_profile, "cpu")

    return {"mode": "cloud-only", "released": True, "reason": "released"}


@app.post("/gpu/release", dependencies=[Depends(require_admin)])
async def gpu_release():
    """Release the GPU model in-process; switch to cloud-only mode.

    External GPU consumers (gpu_guard ConfigConsumer, lerobot, etc.) call
    this endpoint to ask paramem to step aside without exiting. Idempotent:
    a server already in cloud-only returns 200 immediately.

    During an in-flight consolidation cycle the server returns 503; the
    caller may retry. This is a policy choice, not a correctness necessity:
    the cycle re-extracts/resumes cleanly on next start with no persistent
    corruption, but releasing mid-cycle would discard the in-progress work
    of the current cycle, so the refusal avoids that waste.

    On success the response is synchronous — by the time the POST returns,
    the model is unloaded and ``_state["mode"]`` is ``"cloud-only"``. The
    auto-reclaim loop is started so paramem will reclaim the GPU once the
    external consumer goes away (cloud-only → local via service restart,
    same code path as ``--defer-model``).

    Returns:
        200 ``{"mode": "cloud-only", "released": bool, "reason": str}``.
            ``released=False`` when the server was already cloud-only.
        409 ``{"error": "base_swap_active", ...}`` when a base-swap migration
            is actively running (checked first — a swap can transiently hold
            ``_state["mode"] == "cloud-only"`` between its own Phase A → Phase
            B reload, and this door must still refuse rather than take the
            cloud-only idempotent short-circuit).
        503 ``{"error": "consolidating", ...}`` when a cycle is in flight.
    """
    if (_state.get("migration") or {}).get("base_swap_active", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "base_swap_active",
                "message": (
                    "A base-swap migration is actively running. "
                    "Wait for it to complete (or fail) before releasing the GPU."
                ),
            },
        )

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

    result = await _gpu_release_internal()

    # Reclaim-task start is route policy, not the internal's: only a REAL
    # release (the dict shape) starts it — the 503-in-flight JSONResponse
    # above short-circuits before reaching here, and the base-swap
    # orchestration's own fresh-start reload calls `_gpu_release_internal`
    # directly (never this route), so it never starts one either.  See
    # `_gpu_release_internal`'s docstring for why that separation matters.
    if isinstance(result, dict):
        reclaim_task = _state.get("reclaim_task")
        if reclaim_task is None or reclaim_task.done():
            reclaim_interval = _state["config"].server.reclaim_interval_minutes
            _state["reclaim_task"] = asyncio.create_task(_auto_reclaim_loop(reclaim_interval))

    return result


@app.post("/incidents/{incident_id}/ack", dependencies=[Depends(require_admin)])
async def incidents_ack(incident_id: str):
    """Acknowledge an active incident, silencing its attention row.

    Acknowledged incidents remain visible in the incident store but are omitted
    from the loud attention signal in ``/status``.  A subsequent failure of the
    same type reopens the incident (status → active, count bumped), making it
    visible again.  Auto-resolve on the next successful run clears it entirely.

    Args:
        incident_id: Deterministic incident id (``f"{type}:{key}"``), e.g.
            ``"vram_exhausted:phase1"``.  Stable across restarts.

    Returns:
        ``{"status": "ok", "id": incident_id}`` when acknowledged,
        ``{"status": "not_found", "id": incident_id}`` when no matching
        incident exists.
    """
    state_dir = data_state_dir(_state["config"].paths.data)
    ok = ack_incident(state_dir, incident_id)
    return {"status": "ok" if ok else "not_found", "id": incident_id}


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
    admin-scope token minted with ``mint-user-token --scope admin`` (see
    DEPLOYMENT.md — Per-user token management).  The endpoint is
    unreachable in auth-OFF mode (no store configured) and with chat-scope
    tokens, so admin actions are never reachable anonymously regardless of
    ``config.debug``.

    Body: optional ``speaker_id`` query parameter — defaults to the first
    enrolled profile when omitted.  Production always writes session
    jsonls — ``SessionBuffer._append_turn``'s write-and-fsync is
    unconditional — so the on-disk rewrite below always fires; it is not
    gated on ``buffer.debug``.  The binding also flows through the next
    consolidation into adapter weights either way.

    Errors
    ------
    409 ``consolidating`` | ``training_active`` | ``trial_active`` |
    ``cloud_only`` | ``base_swap_active`` | ``consolidation_pending``
        A fold, background training, a migration TRIAL, or an active
        base-swap migration is in flight, the server has no local model
        loaded, or a consolidation event's record is pending resume
        (:func:`active_consolidation`, mapped via :func:`refusal_for`) —
        this door rewrites session jsonls that a pending event's ledger may
        be about to consume.
    """
    verdict = active_consolidation()
    if verdict is not None:
        error, message = refusal_for(
            verdict, doing="re-attributing orphan sessions", then="re-attribute"
        )
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

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
        # Rewrite the on-disk jsonl when one exists.  Every turn is written
        # and fsynced to a per-session jsonl ungated by ``debug``
        # (``SessionBuffer._append_turn``), so ``path.exists()`` is normally
        # True in production too; the guard only skips a session whose
        # transcript already retired (archived or deleted at consolidation)
        # while its in-memory buffer entry survived.  Mode-agnostic.
        path = buffer.session_dir / f"{conv_id}.jsonl"
        if path.exists():
            with open(path, "w") as f:
                for turn in turns:
                    f.write(json.dumps(turn) + "\n")
    logger.info("admin/assign-orphans: %d sessions → speaker %s (%s)", claimed, sname, sid)
    return {"status": "ok", "claimed": claimed, "speaker": sname, "speaker_id": sid}


def _stale_mark_keys(*, config, staled_keys: list[str], label: str, store=None) -> dict:
    """Stale-mark *staled_keys* on disk and settle the resulting RAM state —
    the shared post-erase sequence used by both ``POST /speaker/forget`` and
    ``POST /debug/erase-keys``.

    Delegates to
    :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`, a
    file surgeon that reads every affected tier's registry straight off
    disk. A tier is *affected*, and a key is marked, iff that tier's
    registry holds the key ACTIVE: each such key is withheld — a marker
    reserving the id and carrying no fingerprint, since the active
    fingerprint does not survive the transition — and that tier's bound
    slot manifest is re-stamped so
    :func:`~paramem.adapters.manifest.find_live_slot` still resolves it on
    restart — the single shared implementation for every out-of-fold
    registry-mutation caller. It needs neither a live
    :class:`~paramem.memory.store.MemoryStore` nor a model, which is what
    lets ``POST /debug/erase-keys`` run this sequence in cloud-only mode and
    while the store is quarantined. This is a stale-mark, not a hard erase:
    entries and bookkeeping are untouched, and the row leaves with the rest
    of the key at its owning tier's own next rebuild, when the key is
    genuinely retired rather than merely withheld. A tier's rebuild is a
    full consolidation or ``POST /reconsolidate`` (both rebuild every main
    tier) or an interim cycle (rebuilds only the slot it mints) — a tier no
    consolidation reaches keeps its markers indefinitely. A key already
    withheld (or unknown) in every tier affects nothing — no bytes written, no
    rebind attempted — the ordinary idempotent outcome, not a refusal.

    ``erase_keys_and_restamp_manifest`` NEVER refuses — every affected
    tier's mutation lands and every rebind is attempted regardless of
    outcome, INCLUDING an I/O failure during one tier's own rebind attempt:
    that tier's :class:`~paramem.memory.persistence.RestampResult` carries
    :data:`~paramem.memory.persistence.REBIND_FAILED` (with the caught
    exception's message) rather than aborting the remaining tiers. This
    function reports the per-tier outcome (never raises for it): for every
    affected tier, records the tier's
    :class:`~paramem.memory.persistence.RestampResult` as a
    :class:`~paramem.server.app.TierRestampOutcome` — outcome
    ``"rebound"``, ``"unbound"``, or ``"rebind_failed"`` — and drives that
    same verdict through :func:`_record_or_resolve_tier_health` (rebound ->
    resolves any prior ``tier_registry_unverified`` incident EXCEPT one
    last recorded as
    :data:`~paramem.adapters.registry_binding.PAYLOAD_MISMATCH` — a restamp
    never reads the payload, so it passes ``resolves_payload_status=False``
    and cannot clear that one; unbound OR rebind_failed -> records one,
    plus an ERROR log naming the tier and the reason) — the same incident
    type and record-or-clear implementation the post-fold drift sweep
    (:func:`_record_unverified_tier_incidents`) uses.

    When *store* is given — a caller holding a live, RAM-resident
    ``MemoryStore`` for a healthy (non-quarantined) server —
    ``store.discard_keys(staled_keys)`` is called immediately after the file
    surgery so the RAM registries stay in lockstep with what just landed on
    disk: every named key is unreachable for serving (excluded from
    ``list_active()``) the moment this call returns. *store* is ``None`` for
    a caller with no live store to sync (cloud-only, or the store is
    quarantined) — files remain the driving source and a later lift picks
    them up.

    A no-op (no disk or RAM mutation) when *staled_keys* is empty — returns
    empty ``tiers``/``unbound_tiers``.

    Args:
        config: The live server config the caller re-resolved against
            ``_state["config"]`` (door-staleness safe).
        staled_keys: Keys to stale-mark; sorted, may be empty.
        label: Short caller identifier folded into the malformed-registry
            error detail and the unbound-tier log line — so a
            divergence-repair stale-mark and a speaker forget are
            distinguishable without a second copy of this sequence.
        store: The live :class:`~paramem.memory.store.MemoryStore` to sync
            in RAM after the disk write, or ``None`` when the caller has no
            live store to sync.

    Returns:
        dict with ``staled_keys`` (echoes the input), ``tiers`` (one
        :class:`TierRestampOutcome`-shaped dict per affected tier — always
        200-worthy, the mutation already landed for every one, whatever
        its rebind outcome) and ``unbound_tiers`` (every tier whose
        outcome is not ``"rebound"`` — covers both ``"unbound"`` and
        ``"rebind_failed"`` — also present in ``tiers`` with that outcome).

    Raises:
        HTTPException: 500 ``malformed_tier_name`` when an affected tier's
            on-disk registry file exists but is not KeyRegistry-shaped
            (propagated from
            :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`),
            raised before any mutation.
    """
    tiers_report: list[dict] = []
    unbound_tiers: list[str] = []

    if staled_keys:
        from paramem.adapters.manifest import count_slot_candidates
        from paramem.memory.interim_adapter import iter_tier_roots
        from paramem.memory.persistence import (
            NOTHING_TO_BIND,
            REBIND_FAILED,
            RESTAMPED,
            erase_keys_and_restamp_manifest,
        )

        try:
            results = erase_keys_and_restamp_manifest(
                adapter_dir=config.adapter_dir,
                keys=staled_keys,
            )
        except ValueError as exc:
            # KeyRegistry.load raises ValueError on an existing but
            # non-KeyRegistry-shaped registry file — resolved (and this
            # raised) BEFORE any mutation, so a corrupt tier never leaves
            # the others half-mutated with nothing persisted.
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "malformed_tier_name",
                    "detail": f"{label}: {exc}",
                },
            ) from exc

        tier_roots = dict(iter_tier_roots(config.adapter_dir))
        for tier_name in sorted(results):
            result = results[tier_name]
            rebound = result.status in (RESTAMPED, NOTHING_TO_BIND)
            if rebound:
                outcome = "rebound"
            elif result.status == REBIND_FAILED:
                outcome = "rebind_failed"
            else:
                outcome = "unbound"
            # rebind_failed's reason is the caught exception's message
            # (RestampResult.message); unbound's reason is the planning
            # status restamp_tier_manifest returned — the same two shapes
            # TierRestampOutcome.reason documents.
            reason = None
            if not rebound:
                reason = result.message if outcome == "rebind_failed" else result.status
            tier_root = tier_roots.get(tier_name)
            candidate_count = count_slot_candidates(tier_root) if tier_root is not None else 0
            _record_or_resolve_tier_health(
                config,
                tier=tier_name,
                unhealthy_status=None if rebound else result.status,
                detail=f"{label}: registry mutation landed, manifest re-stamp {result.status}",
                candidate_count=candidate_count,
                # A restamp only proves the registry now binds a slot by
                # hash — it never reads the payload — so a rebound tier's
                # healthy signal here must not clear a payload-level
                # incident the drift sweep recorded (see
                # _record_or_resolve_tier_health's docstring).
                resolves_payload_status=False,
            )
            tiers_report.append(
                {
                    "tier": tier_name,
                    "outcome": outcome,
                    "slot": str(result.slot) if result.slot is not None else None,
                    "reason": reason,
                }
            )
            if not rebound:
                unbound_tiers.append(tier_name)
                logger.error(
                    "%s: tier %s left %s after registry mutation (%s) -- "
                    "recover via a consolidation fold or registry restore",
                    label,
                    tier_name,
                    "REBIND-FAILED" if outcome == "rebind_failed" else "UNBOUND",
                    reason,
                )

        if store is not None:
            store.discard_keys(staled_keys)

    return {
        "staled_keys": staled_keys,
        "tiers": tiers_report,
        "unbound_tiers": unbound_tiers,
    }


@app.post(
    "/speaker/forget",
    response_model=SpeakerForgetResponse,
    dependencies=[Depends(require_admin)],
)
async def speaker_forget(request: SpeakerForgetRequest):
    """Forget a speaker: stale-mark their indexed-memory keys, discard pending sessions.

    The operation performed is a stale-mark, always allowed, never refused:
    a file-surgeon pass
    (:func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`,
    which constructs no :class:`~paramem.memory.store.MemoryStore`) mutates
    every affected tier's on-disk registry — a tier is affected, and a key
    is marked, iff that tier's registry holds the key ACTIVE — and a
    separate :meth:`~paramem.memory.store.MemoryStore.discard_keys` call
    syncs the RAM store to match — safe on a live store, and does not
    trigger retraining, reap nothing, and unmount no adapter. A withheld key
    is immediately unservable (excluded from the tier's ``list_active()``);
    its content and bookkeeping row leave with the rest of the key at its
    owning tier's own next rebuild, when the key is genuinely retired. A
    tier's rebuild is a full consolidation or ``POST /reconsolidate`` (both
    rebuild every main tier) or an interim cycle (rebuilds only the slot it
    mints) — a tier no consolidation reaches keeps its markers indefinitely.
    A key already withheld (or unknown) affects no tier and is an ordinary
    idempotent no-op, not an error.

    Steps
    -----
    1. **Locate the speaker's indexed-memory keys** via the registry bookkeeping
       (``store.iter_bookkeeping()``), which is the source of truth for
       speaker→key and is available between cycles (unlike the transient merged
       graph, which is cleared at cycle-end).

    2. **Stale-mark the keys and re-stamp affected manifests.**
       :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`
       withholds each key found ACTIVE in its owning tier's registry (marker
       only — no fingerprint) and re-stamps that tier's bound slot manifest
       so ``find_live_slot`` still resolves it.

    3. **Remove the speaker profile** from
       :class:`~paramem.server.speaker.SpeakerStore` (persisted immediately).

    4. **Discard any pending sessions** for the speaker from
       :class:`~paramem.server.session_buffer.SessionBuffer`.

    5. **Reload** :attr:`_state`\\ ``["router"]`` so the speaker→key index and
       the simhash-registry cache are fresh after the mutation.

    Steps 3–5 run in a no-await tail after the executor call returns — see
    the handler body for why that atomicity matters.

    Args:
        request: :class:`SpeakerForgetRequest` with ``speaker_id``.

    Returns:
        :class:`SpeakerForgetResponse` reporting what was removed and staled.

    Errors
    ------
    409 ``store_quarantined`` | ``consolidating`` | ``training_active`` | ``trial_active`` |
    ``cloud_only`` | ``base_swap_active`` | ``consolidation_pending``
        The memory store is quarantined (:func:`_store_quarantine_verdict`),
        a fold, background training, a migration TRIAL, or an active
        base-swap migration is in flight, the server has no local model
        loaded, or a consolidation event's record is pending resume
        (mapped via :func:`refusal_for`).  No mutation on any of these.
    500 ``malformed_tier_name``
        An affected tier's on-disk registry file exists but is not
        KeyRegistry-shaped (a ``ValueError`` from ``KeyRegistry.load``,
        propagated from
        :func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`).
        Raised before any mutation and before ``store.discard_keys`` runs
        (from inside the executor — see below), so the store is left
        untouched.
    500 (uncaught)
        Surfaced by FastAPI when a tier's on-disk registry EXISTS but cannot
        be read/decrypted while reading the pre-mutation hash
        (``tier_registry_sha256`` — read BEFORE ``store.discard_keys`` runs,
        so the store is left untouched).
        ``_state["consolidating"]`` is cleared in a ``finally`` regardless of
        outcome.

    Note
    ----
    Discarding an interim slot wholesale (rather than staling one speaker's
    keys within it) is ``POST /interim/discard`` — a separate admin door.
    """
    # Store quarantine is checked first (see _store_quarantine_verdict) —
    # this door mutates the live store, so it must refuse before
    # active_consolidation's own checks.
    verdict = _store_quarantine_verdict() or active_consolidation()
    if verdict is not None:
        error, message = refusal_for(verdict, doing="forgetting a speaker", then="forget")
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

    # Get-or-create rather than "loop is None -> 503": the guard above
    # already proved mode == "local" (a model is loaded), so the loop can
    # always be created here, mirroring POST /interim/discard.
    loop = get_or_create_consolidation_loop(_state)

    # Canonicalize the incoming speaker_id so external cased input
    # (e.g. "Speaker0") matches the internally stored canonical form ("speaker0")
    # produced by set_bookkeeping's is_speaker_id gate.  Single normalization
    # here covers all three comparisons below (bookkeeping, speaker_store.remove,
    # session_meta match).
    speaker_id = request.speaker_id
    if _is_speaker_id(speaker_id):
        speaker_id = _canonical(speaker_id)

    # Locate keys for this speaker via the registry bookkeeping (source of
    # truth) rather than the resident merger.graph (which is cleared at
    # cycle-end and would be empty between cycles).  Pure in-memory read —
    # safe on the event loop, same as /interim/discard's own pure-read
    # inventory step.
    # NOTE: keys minted before speaker_id attribution was introduced carry
    # speaker_id="" in bookkeeping (it was not applied retroactively).  Those
    # keys are a silent-miss here by accepted design — the live setup is for
    # debugging; legacy keys are deliberately not preserved.
    keys: set[str] = {
        key
        for key, record in loop.store.iter_bookkeeping()
        if record.get("speaker_id") == speaker_id
    }
    staled_keys: list[str] = sorted(keys)

    def _forget_sync() -> dict:
        """The disk-touching half of the forget, run off the event loop
        under ``gpu_lock``.  Returns the response fields this half can
        compute (``staled_keys``); the caller's no-await tail fills in
        ``removed_speaker`` and ``discarded_sessions``.
        """
        # Re-resolve rather than close over the handler's pre-lock `loop`
        # AND `config`: a config-apply that won the lock ahead of us may
        # have replaced `_state["config"]` and released + recreated the
        # process-lifetime ConsolidationLoop, and every mutation below
        # (registry writes) must land on the live objects, not a stale
        # pre-lock capture (door staleness race).
        config = _state["config"]
        loop = get_or_create_consolidation_loop(_state)
        return _stale_mark_keys(
            config=config, staled_keys=staled_keys, label="speaker/forget", store=loop.store
        )

    from paramem.server.gpu_lock import gpu_lock

    _state["consolidating"] = True
    try:
        async with gpu_lock():
            loop_aio = asyncio.get_running_loop()
            result = await loop_aio.run_in_executor(None, _forget_sync)

        # NO-AWAIT TAIL — LOAD-BEARING.  From the run_in_executor above returning
        # to the flag clear in `finally` there must be ZERO await points: this
        # coroutine then runs the speaker-store removal, session discard, router
        # reload, and response build without yielding, so no /chat turn can
        # observe a store whose tiers were erased or reaped but whose router
        # index / session buffer still reflect the pre-forget state.  Same
        # argument as `_dispatch_to_executor` / POST /interim/discard's own tail:
        # /chat mutates the session buffer outside gpu_lock (buffer.append) and
        # SessionBuffer has no lock of its own, so this atomicity is what keeps
        # the tail race-free — /chat never reads _state["consolidating"]; gpu_lock
        # alone (acquired above, held across the executor call) keeps /chat off
        # the PEFT mutation.  Adding an await here re-opens both windows at once.
        speaker_store = _state.get("speaker_store")
        removed_speaker = False
        if speaker_store is not None:
            removed_speaker = speaker_store.remove(speaker_id)

        buffer = _state["session_buffer"]
        speaker_conv_ids = [
            conv_id
            for conv_id, session_meta in buffer._sessions.items()
            if session_meta.get("speaker_id") == speaker_id
        ]
        if speaker_conv_ids:
            buffer.discard_sessions(speaker_conv_ids)
        discarded_sessions = speaker_conv_ids

        _state["router"].reload()

        result["removed_speaker"] = removed_speaker
        result["discarded_sessions"] = discarded_sessions

        logger.info(
            "speaker/forget: speaker=%s keys=%d profile_removed=%s sessions=%d",
            speaker_id,
            len(result["staled_keys"]),
            removed_speaker,
            len(discarded_sessions),
        )
        return SpeakerForgetResponse(**result)
    finally:
        _state["consolidating"] = False


def _interim_discard_inventory(loop, config) -> dict:
    """Return the pre-mutation inventory of the interim ring.

    Pure read — touches no state.  One source per fact, no re-derivation:

    - ``store_tiers``: :func:`~paramem.memory.interim_adapter.interim_tiers_newest_first`
      — THE canonical interim-tier enumeration.
    - ``disk_dirs``: the adapter names yielded by
      :func:`~paramem.memory.interim_adapter.iter_interim_dirs` **unfiltered**
      — the exact set :func:`~paramem.memory.interim_adapter.unload_interim_adapters`
      will remove from disk (payload-bearing or not).
    - ``peft_names``: interim-prefixed keys in ``loop.model.peft_config`` when
      ``loop.model`` is resident (not ``None`` — cloud-only), else empty.
    - ``active_keys`` / ``stale_keys``: per-tier counts from the store, for
      every name in ``store_tiers``.

    Returns:
        dict with keys ``"store_tiers"`` (list[str], newest stamp first),
        ``"disk_dirs"`` (list[str], interim adapter names with an on-disk
        directory), ``"disk_dir_names"`` (dict[str, str] mapping each
        ``disk_dirs`` adapter name to its actual on-disk directory name, as
        yielded by ``iter_interim_dirs`` — carried through so a caller never
        has to re-derive the path from the name via
        :func:`~paramem.memory.interim_adapter.interim_dir_for_name`, which
        raises ``ValueError`` on a directory whose stamp suffix isn't
        well-formed (e.g. a stray ``interim_garbage/``); that dir is still a
        real on-disk directory the reaper will remove), ``"peft_names"``
        (list[str]), ``"active_keys"`` (dict[str, int]), ``"stale_keys"``
        (dict[str, int]), and ``"empty"`` (bool — True when ``store_tiers``,
        ``disk_dirs`` and ``peft_names`` are all empty).
    """
    from paramem.memory.interim_adapter import (
        INTERIM_NAME_PREFIX,
        interim_tiers_newest_first,
        iter_interim_dirs,
    )

    store_tiers = interim_tiers_newest_first(loop.store)
    disk_pairs = sorted(iter_interim_dirs(config.adapter_dir), key=lambda pair: pair[0])
    disk_dirs = [name for name, _path in disk_pairs]
    disk_dir_names = {name: path.name for name, path in disk_pairs}
    peft_names = (
        sorted(n for n in loop.model.peft_config if n.startswith(INTERIM_NAME_PREFIX))
        if loop.model is not None
        else []
    )
    return {
        "store_tiers": store_tiers,
        "disk_dirs": disk_dirs,
        "disk_dir_names": disk_dir_names,
        "peft_names": peft_names,
        "active_keys": {t: len(loop.store.active_keys_in_tier(t)) for t in store_tiers},
        "stale_keys": {t: len(loop.store.stale_keys_in_tier(t)) for t in store_tiers},
        "empty": not store_tiers and not disk_dirs and not peft_names,
    }


# The three ring-lifecycle incident types, resolved when the interim ring is
# emptied without a fold (POST /interim/discard).  Single source so a future
# fourth ring incident type is added here, not inlined at the call site.
_RING_LIFECYCLE_INCIDENT_TYPES: tuple[str, ...] = (
    "full_consolidation_overdue",
    "interim_cap_reached",
    "interim_overflow_pending",
)


@app.post(
    "/interim/discard",
    response_model=InterimDiscardResponse,
    dependencies=[Depends(require_admin)],
)
async def interim_discard(request: InterimDiscardRequest):
    """Discard the entire interim ring in-process, without folding it into main memory.

    The only door that removes interim slots without first absorbing their
    content into the main tiers — the full-fold absorb branch
    (:func:`~paramem.training.consolidation.ConsolidationLoop.consolidate`)
    is the other place :func:`~paramem.memory.interim_adapter.unload_interim_adapters`
    is called, and it always runs after the interim keys have been folded
    into the mains.  Here nothing is folded: the facts in every interim slot
    are the only copy, and they are gone once this returns ``"discarded"``.

    Whole-ring only — there is no per-slot selection.  Every downstream
    consequence of the ring (the mint gate's capacity count, the three
    ring-lifecycle incidents, the full-cycle deadline) is ring-level, so a
    partial discard would leave those signals describing a ring that no
    longer matches reality.

    Steps (synchronous; the GPU-touching half runs under ``gpu_lock`` off
    the event loop):

    1. Drop every interim tier from the :class:`~paramem.memory.store.MemoryStore`
       (RAM first — the tier becomes unroutable immediately).
    2. Reap the ring via the one reaper
       (:func:`~paramem.memory.interim_adapter.unload_interim_adapters`):
       delete the PEFT adapters (when any) and ``rmtree`` every on-disk slot,
       payload-bearing or not.
    3. Prune ``loop.promoted_keys`` of the discarded tiers' keys in RAM
       (mirrors ``POST /speaker/forget``'s pruning). Nothing to rewrite on
       disk: the reap in step 2 already deleted each discarded tier's
       directory — key_metadata.json included — wholesale.
    4. Resolve the three ring-lifecycle incidents
       (``full_consolidation_overdue``, ``interim_cap_reached``,
       ``interim_overflow_pending``) — their only other clear site is the
       full-fold absorb path, which this operation deliberately bypasses.
    5. Pop any ``adapter_manifest_status`` rows for the discarded names.
    6. Record the outcome via :func:`~paramem.server.run_status.record_last_run`
       (``op_type="consolidation"``, ``outcome="interim_discarded"``) —
       best-effort; a write failure is logged, not raised, matching every
       other finalizer (e.g. ``_finalize_interim``).
    7. Stamp ``_state["last_consolidation"]`` (on the event loop, inside the
       no-await tail) and reload ``_state["router"]`` so the speaker→key
       index drops the discarded keys.

    Not touched, by design: pending sessions in the ``SessionBuffer`` (never
    in a slot; absorbed by the next fold), main-tier registries/weights,
    donor stores, the schedule stamp.

    Errors
    ------
    409 ``store_quarantined`` | ``consolidating`` | ``training_active`` | ``trial_active`` |
    ``cloud_only`` | ``base_swap_active`` | ``consolidation_pending``
        The memory store is quarantined (:func:`_store_quarantine_verdict`),
        a fold, background training, a migration TRIAL, or an active
        base-swap migration is in flight, the server has no local model
        loaded, or a consolidation event's record is pending resume
        (mapped via :func:`refusal_for`).  No mutation on any of these.
    409 ``confirmation_required`` (``_INTERIM_DISCARD_UNCONFIRMED_STATUS``)
        ``confirm`` was not ``true`` and the ring is non-empty.  The response
        detail carries ``would_discard`` — the same inventory the mutation
        path would destroy — so the operator can see the blast radius before
        committing.  Nothing is mutated.
    500
        Uncaught — surfaced by FastAPI when the reap or the registry rewrite
        fails mid-operation. ``_state["consolidating"]`` is cleared in a
        ``finally`` regardless of outcome; the operation is idempotent and
        safe to retry (a second call reaps whatever the first left behind).
    """
    # Step 0a — shared activity predicate: already-running / cloud-only /
    # bg-training / migration TRIAL / a pending consolidation event's
    # record.  No second implementation of these checks — see
    # active_consolidation's own docstring.  Store quarantine is checked
    # first (see _store_quarantine_verdict) — this door mutates the live
    # store, so it must refuse before active_consolidation's own checks.
    verdict = _store_quarantine_verdict() or active_consolidation()
    if verdict is not None:
        error, message = refusal_for(verdict, doing="discarding the interim ring", then="discard")
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

    config = _state["config"]
    # Step 0b — get-or-create rather than "loop is None -> 503": guard 0a
    # already proved mode == "local" (a model is loaded), so the loop can
    # always be created here, and gating on a pre-existing loop would make
    # this endpoint unusable on a freshly booted server — exactly the state
    # an operator most plausibly wants to discard from.
    loop = get_or_create_consolidation_loop(_state)

    # Step 0c — pure read; a no-op ring must not mutate the store,
    # must not touch incidents, must not log a destructive run.
    inv = _interim_discard_inventory(loop, config)
    if inv["empty"]:
        return InterimDiscardResponse(
            status="noop_empty_ring",
            discarded_tiers=[],
            unloaded_adapters=[],
            removed_dirs=[],
            active_keys_destroyed={},
            stale_keys_destroyed={},
            resolved_incidents=0,
        )

    # Step 0d — the operator sees the blast radius before committing;
    # nothing is written on this path.  Status code is read from
    # _INTERIM_DISCARD_UNCONFIRMED_STATUS, not a hard-coded literal.
    if request.confirm is not True:
        raise HTTPException(
            status_code=_INTERIM_DISCARD_UNCONFIRMED_STATUS,
            detail={
                "error": "confirmation_required",
                "message": (
                    "Discarding the interim ring destroys the only copy of its facts. "
                    'Resend with {"confirm": true} to proceed.'
                ),
                "would_discard": inv,
            },
        )

    from paramem.memory.interim_adapter import unload_interim_adapters
    from paramem.server.gpu_lock import gpu_lock

    state_dir = data_state_dir(config.paths.data)
    # Every name this operation destroys, from every source it could have
    # been recorded under — the union covers a name recorded only via the
    # boot-time disk scan (adapter_manifest_status row with no live PEFT
    # entry) as well as the common case (store + PEFT).
    discarded_names = set(inv["store_tiers"]) | set(inv["peft_names"]) | set(inv["disk_dirs"])
    # Carried through from the inventory (disk_dir_names) rather than
    # re-derived via interim_dir_for_name, which raises ValueError on a
    # stray dir whose stamp suffix isn't well-formed (e.g. interim_garbage/)
    # — a real on-disk directory unload_interim_adapters would still remove.
    removed_dirs = [inv["disk_dir_names"][n] for n in inv["disk_dirs"]]

    def _discard_sync() -> tuple[list[str], int]:
        """Steps 1–6 — the GPU-touching + file-write half, run off the event loop."""
        # Re-resolve rather than close over the handler's pre-lock `loop`
        # AND `config`: a config-apply that won the lock ahead of us may
        # have replaced `_state["config"]` and released + recreated the
        # process-lifetime ConsolidationLoop, and every mutation below
        # (registry drop, PEFT unmount) must land on the live objects, not
        # a stale pre-lock capture (door staleness race).
        config = _state["config"]
        loop = get_or_create_consolidation_loop(_state)

        # Collect every key the discarded interim tiers know (active + stale)
        # before Step 1 drops them — drop_tier removes the tier's registry,
        # so this is the last point the keys are enumerable.
        discarded_keys: set[str] = set()
        for tier in inv["store_tiers"]:
            discarded_keys.update(loop.store.active_keys_in_tier(tier))
            discarded_keys.update(loop.store.stale_keys_in_tier(tier))
        # Step 1 — RAM first: the tier becomes unroutable immediately.
        for tier in inv["store_tiers"]:
            loop.store.drop_tier(tier)
        # Step 2 — ONE reaper, both venues (PEFT delete + on-disk rmtree).
        # The reap removes each discarded tier's directory wholesale,
        # including its own key_metadata.json — there is no surviving
        # per-tier file left to rewrite; nothing else holds a row for a
        # key that lived only in a now-deleted interim tier.
        unloaded = unload_interim_adapters(loop.model, config.adapter_dir)
        # Step 3 — prune promoted_keys of the discarded tiers' keys (mirrors
        # POST /speaker/forget's pruning).
        loop.promoted_keys.difference_update(discarded_keys)
        # Step 4 — the ring's own incidents have no other clear site once
        # the ring is gone (_oldest_interim_stamp returns None post-discard).
        # Same rationale as Step 3: the ring drop already happened, so a
        # bookkeeping failure here must not turn it into an HTTP 500.
        resolved = 0
        try:
            for _type in _RING_LIFECYCLE_INCIDENT_TYPES:
                resolved += resolve_incidents_by_type(
                    state_dir, _type, reason="interim ring discarded without absorption"
                )
        except Exception:
            logger.exception("Post-discard incident resolution failed (non-fatal)")
        # Step 5 — manifest rows describing slots that no longer exist.
        manifest_status = _state.get("adapter_manifest_status", {})
        for name in discarded_names:
            manifest_status.pop(name, None)
        # Step 6 — operator-visible run record; a stale prior-fold row would
        # otherwise misdescribe the post-discard state of memory.  A
        # run_status.json write failure must not turn an already-completed
        # destructive ring drop into an HTTP 500 (matches the six finalizers,
        # e.g. _finalize_interim).
        try:
            record_last_run(
                state_dir,
                op_type="consolidation",
                outcome="interim_discarded",
                summary=(
                    f"Interim ring discarded: {len(inv['store_tiers'])} tier(s), "
                    f"{sum(inv['active_keys'].values())} active key(s)"
                ),
                detail={
                    "discarded_tiers": inv["store_tiers"],
                    "unloaded_adapters": unloaded,
                    "removed_dirs": removed_dirs,
                },
            )
        except Exception:
            logger.exception("Post-discard run-status bookkeeping failed (non-fatal)")
        return unloaded, resolved

    _state["consolidating"] = True
    try:
        async with gpu_lock():
            loop_aio = asyncio.get_running_loop()
            unloaded_adapters, resolved_incidents = await loop_aio.run_in_executor(
                None, _discard_sync
            )

        # NO-AWAIT TAIL — LOAD-BEARING.  From the run_in_executor above returning to the
        # flag clear in `finally` there must be ZERO await points: this coroutine then runs
        # the router reload + response build without yielding, so no /chat turn can observe
        # a store whose interim tiers are gone but whose router index still lists their keys.
        # Same argument as `_dispatch_to_executor`.  Note WHY this is the
        # whole story: /chat never reads _state["consolidating"] — that flag excludes the
        # consolidation arbitrator, nothing else.  /chat is kept off the PEFT mutation by
        # `gpu_lock` alone (acquired above, held across the executor call).  Adding an await
        # here re-opens both windows at once.
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["router"].reload()

        return InterimDiscardResponse(
            status="discarded",
            discarded_tiers=inv["store_tiers"],
            unloaded_adapters=unloaded_adapters,
            removed_dirs=removed_dirs,
            active_keys_destroyed=inv["active_keys"],
            stale_keys_destroyed=inv["stale_keys"],
            resolved_incidents=resolved_incidents,
        )
    finally:
        _state["consolidating"] = False


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

    # get_name is used ONLY for the existence check — it returns the RAW
    # speaker{N} token for an anonymous-enrolled profile (name == id).  That
    # token is no longer a suppression concern for the LOCAL system-prompt
    # identity line: _build_system_prompt keys the "You are speaking with X"
    # line off speaker_id presence directly (anonymous included, per the
    # B-form prefix design), so the raw token IS the identity line for an
    # undisclosed speaker — this is intentional, not a leak, since the
    # speaker is talking to themselves. resolve_speaker_name stays the
    # accessor for the two remaining name-presence-gated surfaces —
    # ChatResponse.speaker below and the greeting salutation elsewhere in
    # this module — both of which return None (not the raw token) for an
    # anonymous profile until it is disclosed.
    if store.get_name(request.speaker_id) is None:
        return JSONResponse(
            {"status": "speaker_not_found", "speaker_id": request.speaker_id},
            status_code=404,
        )
    speaker_name = store.resolve_speaker_name(request.speaker_id)

    # Side-effect-free text-side language detection.  STT detection and
    # tracker.record are /chat-only side-effects and are intentionally omitted.
    from paramem.server import lang_id as _lang_id

    detected_language, _ = _lang_id.resolve_text_language(request.text, config.text_lang_detection)

    # Server-authoritative history — same SessionBuffer read _run_chat_turn
    # uses.  This endpoint never appends (see docstring), so a conversation_id
    # unique to this probe call (the default) always reads back empty; that
    # loss of ad-hoc multi-turn testing is an accepted cost of retiring the
    # client-supplied history field.
    buffer = _state["session_buffer"]
    history = buffer.get_conversation_turns(request.conversation_id)

    # Cloud-only mode mirrors /chat dispatch — no GPU lock, no model.
    if _state["mode"] == "cloud-only":
        cloud_result = _relay_route(
            text=request.text,
            history=history,
            config=config,
            cloud_permitted=(
                _state.get("cloud_only_reason") not in _INVOLUNTARY_CLOUD_ONLY_REASONS
                or config.cloud.allow_degraded_serving
            ),
            ha_client=_state.get("ha_client"),
            cloud_agent=_state.get("cloud_agent"),
            language=detected_language,
            speaker_id=request.speaker_id,
        )
        resolved_text = resolve_speaker_tokens(
            cloud_result.text, store, current_speaker_id=request.speaker_id
        )
        return ChatResponse(text=resolved_text, escalated=True, speaker=speaker_name)

    # Local mode — abort BG trainer + acquire gpu_lock, mirroring /chat.
    _abort_background_training_for_inference()

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
                history=history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=config,
                router=_state["router"],
                cloud_agent=_state.get("cloud_agent"),
                ha_client=_state.get("ha_client"),
                language=detected_language,
                effective_mode=_state.get("effective_mode"),
                memory_store=_state["memory_store"],
            ),
        )

    return ChatResponse(
        text=resolve_speaker_tokens(result.text, store, current_speaker_id=request.speaker_id),
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
    system_prompt: str | None = (
        None  # None → paramem.training.dataset.trained_recall_system_prompt()
    )
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

    available = sorted(model.peft_config.keys())
    if request.adapter != "none" and request.adapter not in available:
        return JSONResponse(
            {
                "status": "unknown_adapter",
                "requested": request.adapter,
                "available": available + ["none"],
            },
            status_code=400,
        )

    _abort_background_training_for_inference()

    from paramem.evaluation.recall import generate_answer
    from paramem.memory.entry import parse_recalled_entry
    from paramem.models.loader import (
        base_model_inference,
        grad_checkpointing_disabled,
        render_chat_prompt,
        switch_adapter,
    )
    from paramem.server.gpu_lock import gpu_lock
    from paramem.training.dataset import trained_recall_system_prompt

    system_prompt = (
        request.system_prompt
        if request.system_prompt is not None
        else trained_recall_system_prompt()
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]
    prompt = render_chat_prompt(messages, tokenizer, add_generation_prompt=True)

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

        # One CM per branch — base_model_inference already disables gradient
        # checkpointing internally (and the active adapter), so wrapping it
        # in grad_checkpointing_disabled too would open the same CM twice.
        t0 = time.monotonic()
        try:
            if request.adapter == "none":
                with base_model_inference(model):
                    raw = generate_answer(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                    )
                adapter_active_label = "disabled"
            else:
                with grad_checkpointing_disabled(model):
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
            if prior:
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

    ``entries``/``total`` reflect the non-authoritative entry mirror
    (``_entries``) only.  Under ``inference.preload_cache=False`` the boot
    fill is skipped, but the mirror's other writer — go-live adoption
    (:meth:`~paramem.memory.store.MemoryStore.adopt_increments`) — installs a
    rebuilt tier's entries regardless of the setting, so the counts are
    empty only until the first fold in the process.
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
    ``inference.preload_cache=false`` the boot fill is skipped, so this
    endpoint returns an empty list until the first consolidation's go-live
    adoption installs entries — a correct read either way, not an error.
    ``bookkeeping_total`` will still be non-zero when the
    router has speaker provenance loaded from ``key_metadata.json``.

    Each entry dict is the entry payload as stored, with ``tier`` and
    ``key`` fields added inline for flat consumption.  Per-key
    ``speaker_id``, ``relation_type``, ``reinforcement_count``,
    ``last_reinforced_cycle``, ``last_seen``, and ``first_seen`` are
    sourced from ``store.bookkeeping_for_key(key)`` (authoritative
    ``_bookkeeping`` dict) rather than the entry payload, which may be
    stale.  Every entry key is registered active (:meth:`MemoryStore.put`
    always registers), so its bookkeeping row is read directly — the
    every-known-key-has-a-row invariant.
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
        # Overlay the FULL authoritative bookkeeping record onto the row. The
        # entry payload (_entries) may carry stale bookkeeping-shaped fields
        # (e.g. speaker_id/relation_type) — _bookkeeping is the single source of
        # truth (store.py:53-58), so it wins. Splatting the whole record (rather
        # than a hand-maintained field list) makes the dump integrally reflect
        # every bookkeeping field, so a newly-added field can never be silently
        # omitted here.
        row.update(store.bookkeeping_for_key(key))
        entries.append(row)
        tiers[tier] = tiers.get(tier, 0) + 1

    return DebugDumpResponse(
        entries=entries,
        total=len(entries),
        tiers=tiers,
        bookkeeping_total=store.bookkeeping_count(),
    )


# --------------------------------------------------------------------------
# Debug erase-keys endpoint — the operator's scalpel for a key that is wrong
# for an unknown reason, when the only other alternatives are a full bundle
# restore (POST /backup/restore) or a full wipe.  A FILE SURGEON: it reads
# the named tiers' persisted registries straight off disk, stale-marks the
# operator's explicit key list, and writes the registry + manifest restamp
# back — no MemoryStore hydration and no model are required, so the door
# works in cloud-only mode and while the store is quarantined.  On a
# quarantined store the file surgery is followed by an attempt at the lift
# (:func:`_lift_quarantined_store`), so a successful repair swaps a fresh
# store in without a restart.  Takes the explicit key list the operator
# already has (from GET /debug/dump output) and stale-marks exactly those —
# no wildcard, no speaker derivation, no "erase everything" mode.
# --------------------------------------------------------------------------


# Status code for POST /debug/erase-keys' unconfirmed refusal — the same
# deliberate 409 reasoning as _INTERIM_DISCARD_UNCONFIRMED_STATUS: named here
# rather than hard-coded so the handler, tests, and docs read one definition.
_DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS: int = 409


class DebugEraseKeysRequest(BaseModel):
    """Request body for ``POST /debug/erase-keys``.

    Attributes
    ----------
    keys:
        Explicit indexed-memory keys to stale-mark.  Production source of
        this value is the operator, reading ``GET /debug/dump`` output —
        there is no wildcard or "all" expansion; the caller names exactly
        the keys to stale-mark.
    confirm:
        Must be ``True`` to actually stale-mark the keys.  ``False`` (default)
        refuses with ``confirmation_required`` — mirrors
        ``POST /interim/discard``'s ``confirm`` field: the operator's own
        invocation is the only source for this value; the system cannot
        derive it.
    """

    keys: list[str]
    confirm: bool = False


class DebugEraseKeysResponse(BaseModel):
    """Response body for ``POST /debug/erase-keys``.

    Attributes
    ----------
    staled:
        Requested keys that were known to an on-disk tier registry (active
        or stale, regardless of bookkeeping presence). A key already
        withheld in every tier that knows it is reported here too, with
        zero mutation — re-erasing it is idempotent, not an error.
    unknown:
        Requested keys no on-disk tier registry recognised — reported,
        never an error; a repeat request with the same list is idempotent.
    lifted:
        ``None`` when the store was not quarantined (no lift was
        attempted — the RAM store, when one exists, was synced directly).
        ``True``/``False`` when the store WAS quarantined at the time of
        this call and the door attempted :func:`_lift_quarantined_store`
        after the file surgery: ``True`` on a successful lift (the store is
        now live and serving again), ``False`` when the lift itself
        re-quarantined (the store step failed again — see
        ``GET /integrity`` and the ``store_quarantined`` incident for the
        updated cause).
    tiers:
        Per-tier :class:`TierRestampOutcome` for every tier the erase
        touched — the registry mutation always landed; this reports
        whether the tier's slot manifest was rebound to match.
    unbound_tiers:
        Tier names left unbound — every ``TierRestampOutcome.outcome !=
        "rebound"``, i.e. ``"unbound"`` or ``"rebind_failed"`` — a non-empty
        list means at least one affected tier will fail to bind on the
        next boot/reload until repaired; see ``GET /integrity`` and the
        ``tier_registry_unverified`` incident it emits for each.
    """

    staled: list[str]
    unknown: list[str]
    lifted: bool | None = None
    tiers: list[TierRestampOutcome]
    unbound_tiers: list[str]


@app.post(
    "/debug/erase-keys",
    response_model=DebugEraseKeysResponse,
    dependencies=[Depends(require_admin)],
)
async def debug_erase_keys(request: DebugEraseKeysRequest):
    """Stale-mark an explicit list of indexed-memory keys — a file surgeon.

    The operator's scalpel: targeted removal of a key that is wrong for an
    unknown reason, when the only other alternatives are a full bundle
    restore (``POST /backup/restore``) or a full wipe — see the module
    comment above this route. Reads the affected tiers' registries straight
    off disk, stale-marks the operator's explicit key list, and writes the
    registry + manifest restamp back
    (:func:`~paramem.memory.persistence.erase_keys_and_restamp_manifest`, via
    the shared :func:`_stale_mark_keys` sequence ``/speaker/forget`` also
    uses) — no :class:`~paramem.memory.store.MemoryStore` hydration and no
    model are required. The only new logic here is partitioning the
    caller's explicit list into known (staled) vs. unknown (reported, not
    an error) by loading every tier's on-disk registry directly — registry
    active ∪ stale, so a key with no bookkeeping record is still
    stale-markable.

    On a QUARANTINED store, the file surgery above is followed by an
    attempt at the lift (:func:`_lift_quarantined_store`): a successful
    repair swaps a fresh store in, clears the marker, resolves the
    incident, and reloads the router — no restart. On a HEALTHY store, the
    live :class:`MemoryStore` (when one is resident) is synced in RAM
    directly and the router is reloaded in place — no lift needed.

    Returns ``forbidden_not_debug`` (403) when ``config.debug=false``.
    Returns ``invalid_keys`` (400) when *request.keys* is empty or contains
    a non-string/empty entry (pydantic's ``list[str]`` typing already
    rejects a non-string body value with 422; this catches the empty-list
    and empty-string cases pydantic's plain type does not).  Refuses with
    409 (:func:`refusal_for`) under every arm :func:`active_consolidation`
    checks EXCEPT cloud-only (``include_cloud_only=False``) — consolidation
    running, background training active, an active migration TRIAL, or an
    active base-swap migration, PLUS its pending-record arm (a stage ledger
    on disk with no fold actively running — ``deferred_event_pending``), so
    this door never mutates tier files a resumable consolidation event
    could republish over. Cloud-only mode does NOT refuse — the door needs
    no resident model — and quarantine does NOT refuse either
    (:func:`_store_quarantine_verdict` is deliberately not composed here;
    see its own docstring) — a quarantined store is exactly the condition
    this door can repair. Refuses with
    :data:`_DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS` (409) ``confirmation_required``
    when ``request.confirm`` is not ``True`` — mirrors ``POST
    /interim/discard``'s confirmation gate: this door mutates on-disk state,
    so nothing is mutated without the operator's explicit confirm.
    """
    config = _state["config"]
    if not getattr(config, "debug", False):
        return JSONResponse({"status": "forbidden_not_debug"}, status_code=403)

    if not request.keys or any(not isinstance(k, str) or not k for k in request.keys):
        return JSONResponse(
            {"status": "invalid_keys", "detail": "keys must be a non-empty list of strings"},
            status_code=400,
        )

    # include_cloud_only=False — the door is a file surgeon and needs no
    # resident model; every other busy/pending arm still applies.  Quarantine
    # is deliberately NOT composed here (see _store_quarantine_verdict's
    # docstring) — a quarantined store is exactly the condition this door
    # can repair.
    verdict = active_consolidation(include_cloud_only=False)
    if verdict is not None:
        error, message = refusal_for(verdict, doing="erasing keys", then="erase")
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

    # The operator sees exactly what they asked to destroy before committing;
    # nothing is written on this path.  Status code is read from
    # _DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS, not a hard-coded literal —
    # mirrors POST /interim/discard's Step 0d.
    if request.confirm is not True:
        raise HTTPException(
            status_code=_DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS,
            detail={
                "error": "confirmation_required",
                "message": (
                    "Staling keys makes them immediately unservable. "
                    'Resend with {"confirm": true} to proceed.'
                ),
                "would_stale": sorted(request.keys),
            },
        )

    def _erase_sync() -> dict:
        # Re-resolve config rather than close over the handler's pre-lock
        # capture — same door-staleness argument as /speaker/forget's
        # _forget_sync. No loop is created here — the door is loop-free.
        config = _state["config"]

        from paramem.memory.interim_adapter import iter_tier_roots
        from paramem.training.key_registry import KeyRegistry

        known: set[str] = set()
        for _tier_name, tier_root in iter_tier_roots(config.adapter_dir):
            known.update(KeyRegistry.load(tier_root / "indexed_key_registry.json").list_known())

        requested = set(request.keys)
        to_stale = sorted(requested & known)
        unknown = sorted(requested - known)

        result = _stale_mark_keys(
            config=config,
            staled_keys=to_stale,
            label="debug/erase-keys",
            store=_state.get("memory_store"),
        )
        result["unknown"] = unknown

        # The store may have been quarantined coming into this call (a
        # resumable-file-only door must keep working in that state). File
        # surgery alone leaves no live store to serve from — attempt the
        # lift so a successful repair swaps a fresh, erase-reflecting store
        # in without a restart.
        result["lifted"] = None
        if to_stale and _state.get("store_quarantine") is not None:
            # Release the process-lifetime ConsolidationLoop holder before
            # lifting — same base by construction as POST /backup/restore's
            # same-base convergence (_restore_converge_sync): without this,
            # _state["consolidation_loop"] keeps its OLD .store (the
            # pre-lift store _lift_quarantined_store replaces), and the next
            # fold's lazily-cached loop would go live against an object the
            # server no longer serves.
            _loop = _state.get("consolidation_loop")
            if _loop is not None:
                try:
                    _loop.release()
                except Exception:
                    logger.exception("Error releasing consolidation loop during erase-keys lift")
            _state["consolidation_loop"] = None
            result["lifted"] = _lift_quarantined_store(config)
        return result

    from paramem.server.gpu_lock import gpu_lock

    _state["consolidating"] = True
    try:
        async with gpu_lock():
            loop_aio = asyncio.get_running_loop()
            result = await loop_aio.run_in_executor(None, _erase_sync)

        # NO-AWAIT TAIL — same atomicity argument as /speaker/forget: from the
        # executor call returning to the flag clear in `finally` there must be
        # ZERO await points, so no /chat turn observes a store whose keys were
        # staled but whose router speaker->key index still lists them. When a
        # lift ran, _lift_quarantined_store already rebuilt the router against
        # the fresh store — a second reload here would just re-derive the
        # identical index from the same object.
        if not result.get("lifted"):
            _state["router"].reload()

        logger.info(
            "debug/erase-keys: staled=%d unknown=%d lifted=%s",
            len(result["staled_keys"]),
            len(result["unknown"]),
            result["lifted"],
        )

        return DebugEraseKeysResponse(
            staled=result["staled_keys"],
            unknown=result["unknown"],
            lifted=result["lifted"],
            tiers=result["tiers"],
            unbound_tiers=result["unbound_tiers"],
        )
    finally:
        _state["consolidating"] = False


# --------------------------------------------------------------------------
# Calibration endpoints — opt-in dev tool for live prompt iteration.
# Gated by consolidation.calibrate_endpoint_enabled (default False).
# Each endpoint is a thin wrapper around the existing pipeline helper for
# that stage — injection of prompts/params, capture of output.  Stages
# are stop points: the calibration client chains them, and "skip stage X"
# means don't call X's endpoint.  No call modifies weights or writes
# production data on disk.  See paramem/server/calibrate.py.
# --------------------------------------------------------------------------


def _mint_run_identity(route_path: str) -> "tuple[str, Path]":
    """Mint one calibration run's identity — the one place ``run_id`` and
    ``artifact_dir`` are minted, for every ``/calibrate/*`` route.

    ``run_id`` is :func:`~paramem.utils.artifacts.run_stamp`, minted once;
    ``artifact_dir`` is :func:`~paramem.utils.artifacts.artifact_run_dir`
    resolved against that same stamp — the two must never be minted
    separately, or a run's directory and its wire ``run_id`` could drift
    apart.
    """
    config = _state["config"]
    run_id = run_stamp()
    artifact_dir = artifact_run_dir(config.paths.calibration_artifacts, route_path, run_id)
    return run_id, artifact_dir


def _submit_spec(
    spec: "calibrate_module.CalibrationRunSpec", *, action: ConsolidationAction
) -> ConsolidateResponse:
    """Submit an already-built calibration run spec through the arbitrator —
    the shared tail behind every ``/calibrate/*`` route, called after the
    route's own preflight, validate, and spec construction have already
    run on the event loop.

    ``_state["calibration_run"]`` is published only on
    ``status == "started_calibration"`` — the exact string, never the
    ``started_*`` prefix — so neither a deferred/noop dispatch nor a
    migration pre-empt (``started_migration``) overwrites the record of a
    run that was actually submitted.  One slot, not a ring: it holds the
    latest started run and, once its terminal fires
    (:func:`_run_calibration_sync`), its outcome.

    Args:
        spec: The run's validated :class:`~paramem.server.calibrate.CalibrationRunSpec`,
            identity already minted.
        action: ``CALIBRATE`` (every route whose artifact the operator
            supplies) or ``CALIBRATE_PENDING`` (``/calibrate/extract_pending``).

    Returns:
        :class:`ConsolidateResponse` — ``run_id``/``artifact_dir`` present
        exactly when ``status == "started_calibration"``.
    """
    status, resolved_action = _dispatch_consolidation(action, spec=spec)
    if status == "started_calibration":
        _state["calibration_run"] = {
            "run_id": spec.run_id,
            "action": resolved_action.value,
            "route": spec.route_path,
            "artifact_dir": str(spec.artifact_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "outcome": None,
            "finished_at": None,
        }
        return ConsolidateResponse(
            status=status,
            action=resolved_action.value,
            run_id=spec.run_id,
            artifact_dir=str(spec.artifact_dir),
        )
    return ConsolidateResponse(status=status, action=resolved_action.value)


def _submit_calibration_run(
    stage: str,
    req: Any,
    *,
    action: ConsolidationAction = ConsolidationAction.CALIBRATE,
) -> ConsolidateResponse:
    """Mint a declared calibration run's identity, build its spec, and
    submit it — the boundary tail shared by the nine declared
    ``/calibrate/*`` stages (the five chain routes via
    :data:`~paramem.server.calibrate._CHAIN`, the four standalone routes
    via :data:`~paramem.server.calibrate._STANDALONE`), called after the
    route's own preflight has already run on the event loop.

    Mints ``run_id``/``artifact_dir`` via :func:`_mint_run_identity`
    (the one minting site every ``/calibrate/*`` route uses, this one and
    ``/calibrate/extract_pending``'s route alike), builds the spec
    via :func:`~paramem.server.calibrate.build_spec` (which runs the
    stage's own ``validate_*`` as a side effect — zero inference cost, so
    a 400 costs nothing), and submits through :func:`_submit_spec`.
    ``/calibrate/extract_pending`` is not one of the nine — its dispatch
    closure is route-specific (see
    :func:`~paramem.server.calibrate.validate_extract_pending`'s
    docstring) — so its route builds a spec directly and calls
    :func:`_submit_spec` itself.

    Args:
        stage: A key of :data:`~paramem.server.calibrate._CHAIN` or
            :data:`~paramem.server.calibrate._STANDALONE`.
        req: The stage's own request model.
        action: ``CALIBRATE`` (default) — every declared stage answers
            this action; ``CALIBRATE_PENDING`` belongs only to
            ``extract_pending``, which does not call this function.

    Returns:
        :class:`ConsolidateResponse` — ``run_id``/``artifact_dir`` present
        exactly when ``status == "started_calibration"``.
    """
    route_path = calibrate_module.route_path_for(stage)
    run_id, artifact_dir = _mint_run_identity(route_path)
    spec = calibrate_module.build_spec(stage, _state, req, run_id=run_id, artifact_dir=artifact_dir)
    return _submit_spec(spec, action=action)


def _dispatch_calibrate_chain(use_case: str, req: "calibrate_module.CalibrateChainRequest"):
    """Boundary + submit for the five chain-endpoint routes.

    Runs :func:`~paramem.server.calibrate.preflight` on the event loop,
    then submits through :func:`_submit_calibration_run` — the one handler
    behind every chain endpoint.  The route name selects which calibration
    use case's declaration (start step, injected artifact, stop step)
    applies (see :data:`~paramem.server.calibrate._CHAIN`); validation
    itself now runs inside :func:`~paramem.server.calibrate.build_spec`.
    """
    calibrate_module.preflight(_state)
    return _submit_calibration_run(use_case, req)


@app.post(
    "/calibrate/extract",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_extract_route(
    req: calibrate_module.CalibrateChainRequest,
) -> ConsolidateResponse:
    return _dispatch_calibrate_chain("extract", req)


@app.post(
    "/calibrate/procedural",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_procedural_route(
    req: calibrate_module.CalibrateChainRequest,
) -> ConsolidateResponse:
    return _dispatch_calibrate_chain("procedural", req)


@app.post(
    "/calibrate/anonymize",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_anonymize_route(
    req: calibrate_module.CalibrateChainRequest,
) -> ConsolidateResponse:
    return _dispatch_calibrate_chain("anonymize", req)


@app.post(
    "/calibrate/plausibility",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_plausibility_route(
    req: calibrate_module.CalibrateChainRequest,
) -> ConsolidateResponse:
    return _dispatch_calibrate_chain("plausibility", req)


@app.post(
    "/calibrate/enrich",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_enrich_route(
    req: calibrate_module.CalibrateChainRequest,
) -> ConsolidateResponse:
    return _dispatch_calibrate_chain("enrich", req)


@app.post(
    "/calibrate/normalize",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_normalize_route(
    req: calibrate_module.CalibrateNormalizeRequest,
) -> ConsolidateResponse:
    calibrate_module.preflight(_state)
    return _submit_calibration_run("normalize", req)


@app.post(
    "/calibrate/anonymize_facts",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_anonymize_facts_route(
    req: calibrate_module.CalibrateAnonymizeFactsRequest,
) -> ConsolidateResponse:
    calibrate_module.preflight(_state)
    return _submit_calibration_run("anonymize_facts", req)


@app.post(
    "/calibrate/name",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_name_route(req: calibrate_module.CalibrateNameRequest) -> ConsolidateResponse:
    calibrate_module.preflight(_state)
    return _submit_calibration_run("name", req)


@app.post(
    "/calibrate/respond",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_respond_route(
    req: calibrate_module.CalibrateRespondRequest,
) -> ConsolidateResponse:
    """Run one production serving turn through ``handle_chat`` for calibration.

    Runs the training-abort sequence on the event loop, before dispatch —
    matching every other inference entry point in this module — but does
    NOT stamp ``_state["last_chat_monotonic"]``.  That marker exists so a
    fold does not seize the GPU seconds after a LIVE user turn; this route
    is a calibration probe of the serving path, not a live turn, and
    stamping it here would make the arbitrator's own idle debounce defer
    this call against itself (elapsed time since the stamp is always ~0s).
    This run still holds the ``consolidating`` mutex for the whole turn,
    network call included — the run may reach Home Assistant or place a
    billed cloud call.
    """
    _abort_background_training_for_inference()
    calibrate_module.preflight(_state)
    return _submit_calibration_run("respond", req)


@app.post(
    "/calibrate/extract_pending",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin)],
)
async def calibrate_extract_pending_route(
    req: calibrate_module.CalibrateExtractPendingRequest,
) -> ConsolidateResponse:
    """Run the whole session-tier extraction chain over the pending NAMED
    session set — exactly what a fold would take right now.

    Non-staging: stages no event, mints no key, retires no session, writes
    no production incident or attention row.  Noops
    (``noop_no_pending``/``noop_no_named``) exactly as ``/consolidate/interim``
    does, since its content is the identical pending NAMED set.
    """
    calibrate_module.preflight(_state)
    resolved = calibrate_module.validate_extract_pending(_state, req)

    def _dispatch() -> "tuple[Any, dict]":
        loop = get_or_create_consolidation_loop(_state)
        extraction = _extract_pending_sessions(loop, lock_held=True)
        parsed = {
            "sessions": extraction.per_session,
            "episodic_rels": len(extraction.episodic_rels),
            "procedural_rels": len(extraction.procedural_rels),
            "aborted": extraction.aborted is not None,
        }
        return "", parsed

    route_path = "/calibrate/extract_pending"
    run_id, artifact_dir = _mint_run_identity(route_path)
    spec = calibrate_module.CalibrationRunSpec(
        stage="extract_pending",
        route_path=route_path,
        run_id=run_id,
        artifact_dir=artifact_dir,
        dispatch=_dispatch,
        # Every session this run reaches opens a "local_extract" phase; the
        # gate names the whole-chain run's own step, not one session's.
        input_prompt_phase="local_extract",
        supports_seed=True,
        params=req.params,
        overrides=resolved["overrides"],
    )
    return _submit_spec(spec, action=ConsolidationAction.CALIBRATE_PENDING)


def _run_calibration_sync(spec: "calibrate_module.CalibrationRunSpec") -> None:
    """Execute one calibration run under the shared envelope.

    Executor entry point.  Opens the run's artifact root INSIDE this frame
    (``run_in_executor`` does not carry the caller's ContextVars, so the
    scope must be opened on the worker thread), applies the operator's
    prompt overrides for the whole run, evicts voice when the run's own
    artifact is document-shaped, takes the GPU cooldown gate and the GPU
    lock, pins cuDNN determinism, calls
    :func:`~paramem.server.calibrate.run_stage`, writes ``response.json``,
    and exits through :func:`_consolidation_terminal`.  This is the one
    owner of the ``calibration_run``/``prompt_overrides`` scope for a real
    dispatch — nothing else opens it.

    Stages no event: no PEFT slot, no ``stage_event``, no stage ledger, no
    training, no session retirement, no production incident or attention
    write.
    """
    from paramem.server.gpu_lock import gpu_lock_sync

    config = _state["config"]
    wait_for_cooldown(
        config.vram.cooldown_gate_threshold_c,
        config.vram.cooldown_gate_max_wait_fold_s,
        config.vram.cooldown_gate_poll_s,
        label="fold",
    )
    with calibration_run(spec.artifact_dir), prompt_overrides(spec.overrides):
        if spec.evicts_voice:
            _set_voice_pipeline_profile("cpu", lock_held=False)
        with gpu_lock_sync(), calibrate_module._cudnn_deterministic():
            payload = calibrate_module.run_stage(spec, _state)
        _end_voice_eviction(lock_held=False)
        on_calibration_result(payload, stamp=spec.run_id)

    def _terminal() -> None:
        outcome = "unreached_step" if payload.get("unreached_step") else "completed"
        try:
            record_last_run(
                data_state_dir(config.paths.data),
                op_type="calibration",
                outcome=outcome,
                summary=f"Calibration {spec.stage}: {outcome}",
                detail={"route": spec.route_path, "run_id": spec.run_id},
            )
        except Exception:
            logger.exception("Failed to record calibration run status (non-fatal)")
        _record = _state.get("calibration_run")
        if _record is not None and _record.get("run_id") == spec.run_id:
            _record["outcome"] = outcome
            _record["finished_at"] = datetime.now(timezone.utc).isoformat()

    _consolidation_terminal(_terminal)


def _trial_active() -> bool:
    """True when a migration TRIAL is in progress and consolidation must refuse.

    Thin wrapper around :func:`paramem.server.trial_state.trial_active` bound
    to this module's own ``_state`` — the single predicate shared by both
    refusals that must never drift apart: :func:`require_no_trial` (the
    FastAPI dependency — HTTP 409 on every consolidation route and on
    ``/ingest-sessions``' in-handler gate) and :func:`_consolidation_dispatch_guards`
    (the arbitrator's own guard, ``"deferred_trial_active"``, for in-process
    callers that never go through FastAPI's dependency resolution).

    During a TRIAL the candidate store is live but unaccepted; starting a
    consolidation run would train against a store the operator may still roll
    back.
    """
    return trial_active(_state)


async def require_no_trial() -> None:
    """FastAPI dependency: refuse consolidation while a migration TRIAL is active.

    Thin wrapper around :func:`_trial_active` — see its docstring for what
    "trial active" means and why the predicate is shared with the
    arbitrator's own guard. Every consolidation door carries this guard, so
    the refusal is identical whoever knocks.

    Applied to the four consolidation routes only.  It is deliberately NOT a
    router-wide dependency: FastAPI resolves dependencies BEFORE request-body
    validation, so on a body-carrying route it would turn a malformed body's
    422 into a 409.  None of the ten ``/calibrate/*`` routes carry it for
    exactly that reason — they all carry bodies — so a TRIAL in progress
    produces 200 ``deferred_trial_active`` from the arbitrator's own guard
    (:func:`_consolidation_dispatch_guards`) on those routes instead of a
    409 here.  ``/ingest-sessions`` keeps its own in-handler guard for the
    same reason.

    Raises:
        HTTPException 409 ``trial_active``: A migration TRIAL is in progress.
    """
    if _trial_active():
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


@app.post(
    "/scheduled-tick",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin), Depends(require_no_trial)],
)
async def scheduled_tick():
    """Systemd user-timer entrypoint (paramem-consolidate.timer).

    Requests ``AUTO``. The only other ``AUTO`` requester is the boot-completion
    catch-up task (:func:`_run_boot_completion_tasks`), which dispatches
    in-process rather than through this route — this is the only REST door
    that does. ``AUTO`` carries the schedule's own bookkeeping: the
    suspend/power-off catch-up gate (a tick that is not yet due against its
    own cadence mark fires no cycle — the same universal gate applies whether
    the request reaches here or the boot-completion task), the deadline
    resolution (:func:`_is_full_cycle_due` picking a full fold or an interim
    cycle), and the cadence stamp (:func:`_stamp_scheduled_run`) on dispatch.
    ``action`` in the response reports which of the two was resolved.  The
    content gate that follows the resolution is the same one
    ``POST /consolidate`` and ``POST /consolidate/interim`` are subject to —
    it is not exclusive to the timer.

    Non-blocking; HTTP 200 in every case, read ``status``.  If the GPU is
    unavailable (cloud-only, or bg training already active) the status is
    ``deferred_*`` — the timer will fire again on its next wall-clock tick.

    Takes no request body.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress
    ("refuses new cycles" while TRIAL is active).
    """
    status, action = _dispatch_consolidation(ConsolidationAction.AUTO)
    return ConsolidateResponse(status=status, action=action.value)


@app.post(
    "/consolidate",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin), Depends(require_no_trial)],
)
async def consolidate():
    """Collapse the recent conversations into main memory now — content-gated,
    not time-gated.

    Requests ``FULL`` directly: it is gated by the same content check the
    schedule uses to decide a full cycle is due
    (:func:`~paramem.server.consolidation_action.consolidation_content_gate`'s
    CONTENT check — payload-bearing interim slots, or at ``max_interim_count: 0``
    pending NAMED sessions), minus the schedule's own deadline math
    (:func:`_is_full_cycle_due`), which is the schedule's business alone and is
    never consulted on this path.  It never resolves ``AUTO`` and never falls
    back to an interim absorb: if there is nothing to fold, it noops.

    A manual request drops only the TIME condition (is a full cycle due
    *right now*); the CONTENT condition (is there anything for it to
    consume) still applies, via the identical content gate
    ``POST /scheduled-tick`` uses when it resolves ``FULL``.  With a
    content-bearing interim slot on disk, every interim slot is folded into
    the main tiers, which are re-groomed and re-learned, and the absorbed
    slots are reaped — this absorbs slots left stranded by a later
    ``max_interim_count`` reduction to 0 as well, since the slot check does
    not depend on the CURRENT count.  To absorb only the recent conversations
    without touching main memory, use ``POST /consolidate/interim``; to
    rebuild main memory from its own stored knowledge — turned away only by
    an empty store, never by the absence of new interim/pending material —
    use ``POST /reconsolidate``.

    Takes no request body.

    Non-blocking: the run is submitted to an executor and this returns at once.
    Poll ``GET /status`` (``consolidating``).  Every outcome below is HTTP 200 —
    read ``status``:

    - ``started_full`` — submitted.
    - ``noop_no_interim_slots`` — no content-bearing interim slot (and
      ``max_interim_count > 0``, so pending sessions are not this fold's
      input either).
    - ``noop_no_pending`` / ``noop_no_named`` — ``max_interim_count == 0``
      (no interim slots ever minted) and no pending NAMED session either.
    - ``deferred_*`` — busy (a run is already going, someone is chatting, the
      GPU is held, or the server is cloud-only).  Retry later.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress.
    """
    status, action = _dispatch_consolidation(ConsolidationAction.FULL)
    return ConsolidateResponse(status=status, action=action.value)


@app.post(
    "/consolidate/interim",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin), Depends(require_no_trial)],
)
async def consolidate_interim():
    """Absorb recent conversations into memory now, without waiting for the schedule.

    Extracts every attributable conversation that is still pending and learns it
    into memory.  Use it when something was just said that the assistant should
    know immediately.  Main memory is untouched — that is what
    ``POST /consolidate`` and ``POST /reconsolidate`` are for.

    Requests ``INTERIM`` directly — the same content check the schedule uses
    when it resolves an interim absorb (pending NAMED sessions), minus the
    deadline math that decides WHETHER the schedule would pick interim over
    full.  A manual request drops only that TIME condition: the CONTENT
    condition still applies — a call with zero pending sessions has nothing
    to extract or train, and noops rather than starting an empty cycle.

    Takes no request body.

    Returns (HTTP 200 in every case below; read ``status``):

    - ``started`` — the run was submitted; poll ``GET /status`` (``consolidating``).
    - ``noop_no_pending`` / ``noop_no_named`` — no pending sessions (or none
      attributable) to extract.
    - ``noop_no_interim_tier`` — the deployment is configured with
      ``consolidation.max_interim_count: 0``, so recent conversations are never
      staged separately; they are absorbed by the scheduled consolidation
      itself.  Use ``POST /consolidate`` there.
    - ``deferred_*`` — busy (a run is already going, someone is chatting, the
      GPU is held, or the server is cloud-only).  Retry later.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress.
    """
    status, action = _dispatch_consolidation(ConsolidationAction.INTERIM)
    return ConsolidateResponse(status=status, action=action.value)


@app.post(
    "/reconsolidate",
    response_model=ConsolidateResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_admin), Depends(require_no_trial)],
)
async def reconsolidate():
    """Rebuild main memory from its own stored knowledge — even when nothing is new.

    This is the operation to run after changing the extraction prompts or the
    extraction config: main memory is reconstructed from the keys it already
    holds, re-groomed, and re-learned from the result.  A model change is a
    different flow — the base-swap active-store migration, not this door.

    A reconcile IS a full consolidation whose input excludes pending
    sessions: one fold topology throughout — the interim ring is recalled,
    absorbed into the main tiers, and reaped, exactly as any full fold; warm
    start is uniform, with no cold-start arm.  Only pending sessions differ:
    they stay pending here, unlike an ordinary full fold at
    ``max_interim_count == 0``.  The pending conversations are still there
    for ``POST /consolidate`` or the schedule to absorb afterwards.

    This door has no relationship to a pending consolidation event's
    stage-ledger record: it never discards one, and it is not a recovery or
    abandon door.  A pending interrupted run is resumed and finished first,
    exactly like the other three consolidation endpoints — see
    :func:`_dispatch_consolidation`'s resume-pending-first step — and this
    rebuild request waits for the next dispatch.  A record stuck in a
    deterministic resume-failure loop has exactly one in-band exit: restoring
    a healthy snapshot bundle (``POST /backup/restore``), whose wholesale
    tier rewrite discards the record as part of the restore.

    Its input is the knowledge already stored — every active key in any
    registered tier, main or interim — so it is turned away only by an
    empty store, never by the absence of the NEW material ``POST /consolidate``
    and ``POST /consolidate/interim`` require: a call with no interim slot
    and no pending session still dispatches here.  It noops
    (``noop_no_stored_keys``) only when the store itself holds no active key
    in any tier — nothing to rebuild.  The run does not move the cadence
    window.  It still passes through the shared safety guards ahead of the
    gate — busy/cloud-only/bg-training
    (``_consolidation_dispatch_guards``), a main tier's registry binding
    being unverified, the idle debounce, and the active-store migration
    pre-empt — so a busy server still defers it.

    Takes no request body.

    Returns (HTTP 200; read ``status``):

    - ``started_full`` — the rebuild was submitted; poll ``GET /status``
      (``consolidating``).  It seizes the GPU for the duration.
    - ``noop_no_stored_keys`` — no tier holds an active key; there is
      nothing to rebuild.
    - ``deferred_*`` — busy (a run is already going, someone is chatting, the
      GPU is held, or the server is cloud-only).  Retry later.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress.
    """
    # Bare dispatch.  A pending interrupted run is resumed and finished
    # first (`_dispatch_consolidation`'s resume-pending-first step) — the
    # identical contract the other three consolidation endpoints already
    # have.  This door never discards a pending event's record.
    status, action = _dispatch_consolidation(ConsolidationAction.RECONCILE)
    return ConsolidateResponse(status=status, action=action.value)


# --- Document ingest endpoints ---


@app.post(
    "/ingest-sessions",
    response_model=IngestSessionsResponse,
    dependencies=[Depends(require_admin)],
)
async def ingest_sessions(request: IngestSessionsRequest):
    """Queue pre-chunked document segments for consolidation.

    All chunks in one request belong to a single document and share a
    ``doc_id`` (``"doc-" + secrets.token_hex(4)``).  Each chunk gets its
    own session id ``<doc_id>-c<chunk_index:03d>``.  Re-ingesting the same
    content always queues fresh sessions — idempotency at the content level
    is handled by the graph merger (which is idempotent on
    ``(subject, predicate, object)``), not by this endpoint.

    The original document bytes (``document_b64``) are decoded and stored
    as ``<doc_id>.origdoc`` in ``session_dir`` via
    :meth:`SessionBuffer.write_origdoc`.  The file is archived or deleted
    together with the chunk JSONLs when the document retires via
    :meth:`SessionBuffer.mark_consolidated`.

    Errors
    ------
    400
        ``speaker_id`` is an empty string, OR decoded ``document_b64``
        exceeds 25 MiB (detail ``{"error":"document_too_large"}``).
    404
        ``speaker_id`` not found in ``SpeakerStore``.
    409
        A migration TRIAL is in progress.

    Args:
        request: Payload containing ``speaker_id``, chunk list,
            ``document_filename``, and ``document_b64``.

    Returns:
        :class:`IngestSessionsResponse` with session IDs enqueued,
        ``doc_id``, and rejection flags.
    """
    import base64

    total_chunks = len(request.sessions)

    # Gate 1: empty speaker_id
    if not request.speaker_id:
        return JSONResponse(
            status_code=400,
            content=IngestSessionsResponse(
                queued=[],
                total_chunks=total_chunks,
                rejected_no_speaker_id=True,
            ).model_dump(),
        )

    # Gate 2: document_b64 size guard (decode to check size; keep bytes for storage)
    try:
        doc_raw_bytes = base64.b64decode(request.document_b64)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "document_b64_invalid"},
        )
    if len(doc_raw_bytes) > _ORIGDOC_MAX_BYTES:
        return JSONResponse(
            status_code=400,
            content={"error": "document_too_large"},
        )

    # Gate 3: unknown speaker
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
                rejected_unknown_speaker=True,
            ).model_dump(),
        )

    # Gate 4: migration TRIAL in progress
    if _trial_active():
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

    buffer: SessionBuffer = _state["session_buffer"]

    # One doc_id per request; chunk sessions use <doc_id>-c<chunk_index:03d>.
    doc_id = "doc-" + secrets.token_hex(4)
    chunk_count = len(request.sessions)
    queued: list[str] = []

    for chunk in request.sessions:
        session_id = f"{doc_id}-c{chunk.chunk_index:03d}"

        # set_speaker must precede append so get_pending finds speaker_id.
        buffer.set_speaker(session_id, request.speaker_id, speaker_name or "")
        buffer.set_document_metadata(session_id, doc_id=doc_id, chunk_count=chunk_count)
        buffer.append_document_chunk(
            session_id,
            "user",
            chunk.chunk,
            embedding=None,
            # Turn metadata carries doc_id, chunk_count, and doc_filename so
            # they survive disk serialisation and can be recovered by
            # rehydrate_from_disk after a server restart.
            metadata={
                "source_type": "document",
                "doc_title": chunk.doc_title,
                "chunk_index": chunk.chunk_index,
                "source_path": chunk.source,
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "doc_filename": request.document_filename,
            },
        )
        queued.append(session_id)

    # Store the original document bytes once for the whole request.
    buffer.write_origdoc(doc_id, doc_raw_bytes)

    return IngestSessionsResponse(
        queued=queued,
        total_chunks=total_chunks,
        doc_id=doc_id,
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
    The ``<doc_id>.origdoc`` blob is removed by ``discard_sessions`` when
    all chunk sessions for a document are being discarded.

    Args:
        request: Payload with a list of session IDs to cancel.

    Returns:
        :class:`IngestCancelResponse` splitting the requested IDs into
        ``cancelled`` (found and removed) and ``not_found`` (unknown).

    Errors
    ------
    409 ``consolidating`` | ``training_active`` | ``trial_active`` |
    ``cloud_only`` | ``base_swap_active`` | ``consolidation_pending``
        A fold, background training, a migration TRIAL, or an active
        base-swap migration is in flight, the server has no local model
        loaded, or a consolidation event's record is pending resume
        (:func:`active_consolidation`, mapped via :func:`refusal_for`) —
        under a pending record these sessions are already extracted into
        the ledger and will be trained by the resume, so a 200 here would
        report a cancellation that did not happen.
    """
    verdict = active_consolidation()
    if verdict is not None:
        error, message = refusal_for(
            verdict, doing="cancelling queued ingest sessions", then="cancel"
        )
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

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

    A pre-flight failure (disk pressure, or the check itself raising) does
    NOT raise — it returns 200 with ``pre_flight_fail`` set
    (``"disk_pressure"`` or ``"check_error"``) and ``state`` left at
    ``"LIVE"``; no stash is persisted.
    """
    from fastapi import HTTPException

    from paramem.server.migration import (
        CandidateConfigInvalid,
        _parse_candidate,
        _sha256_bytes,
        compute_shape_changes,
        compute_tier_diff,
        compute_unified_diff,
        detect_simulate_mode,
        initial_migration_state,
        render_preview_response,
        validate_candidate,
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
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
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
    # The stash keeps ${VAR} templates verbatim, so the parse stage stays
    # substitution-free.  Construction runs separately below and its result is
    # discarded — it must never reach the stash or a diff.
    try:
        parsed_candidate = _parse_candidate(candidate_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "candidate_unparseable", "message": str(exc)},
        ) from exc

    # --- Construct the candidate as if it already sat at the live config path ---
    # Parseable YAML is not a bootable config.  Reject here, at LIVE, so an
    # unbootable candidate is never staged: state stays LIVE, nothing is stashed.
    # The returned ServerConfig is discarded by every OTHER caller of
    # validate_candidate (it carries interpolated secrets), but this one is
    # kept — compute_shape_changes below reads tier existence from its
    # tier_config_map() rather than re-deriving adapters.<tier>.enabled from
    # the raw candidate dict a second time.
    try:
        candidate_config = validate_candidate(candidate_bytes, live_config_path)
    except CandidateConfigInvalid as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "candidate_invalid_config", "message": str(exc)},
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
    adapter_dir = config.adapter_dir if config is not None else default_data_dir() / "adapters"
    shape_changes, shape_change_warnings = compute_shape_changes(
        parsed_candidate, adapter_dir, candidate_config
    )

    # --- Detect simulate-mode ---
    simulate_mode_override = detect_simulate_mode(parsed_candidate)

    # --- Pre-flight: disk-pressure gate on backup store ---
    # This endpoint owns the check_error mint below (compute_pre_flight_check
    # raises rather than faking a pass; the except branch here catches that).
    from paramem.backup.preflight import PreFlightCheck
    from paramem.backup.preflight import compute_pre_flight_check as _compute_pre_flight
    from paramem.server.migration import MigrationStashState

    _adapter_dir_for_pf = None
    try:
        if config is not None and hasattr(config, "paths") and config.paths.data is not None:
            _adapter_dir_for_pf = config.adapter_dir
    except (AttributeError, TypeError):
        _adapter_dir_for_pf = None

    try:
        _backups_root_for_pf = (config.paths.data / "backups").resolve()
    except (AttributeError, TypeError):
        _backups_root_for_pf = (default_data_dir() / "backups").resolve()

    try:
        pre_flight = _compute_pre_flight(
            server_config=config,
            loop=_state.get("consolidation_loop"),
            backups_root=_backups_root_for_pf,
            live_config_path=live_config_path,
            adapter_dir=_adapter_dir_for_pf,
        )
    except Exception:
        # An exception here must not surface as an uncaught 500 — but it must
        # not silently render as "ran and passed" either (that was
        # indistinguishable from `pre_flight_fail: None`, the honest-pass
        # case, until this fix). Log it and take the SAME fail branch a real
        # disk_pressure check would take: state stays LIVE, no stash is
        # persisted, and the response carries a distinct fail code the CLI's
        # forward-compat "unknown code" branch already turns into rc=1.
        logger.exception("migration_preview: pre-flight check raised — treating as check_error")
        # No real measurement was taken — None rather than 0, which would
        # read as "store is empty" once rendered.  PreFlightCheck declares
        # this shape (fail_code="check_error" pairs with None measurement
        # fields) — see its docstring.
        pre_flight = PreFlightCheck(
            fail_code="check_error",
            disk_used_bytes=None,
            disk_cap_bytes=None,
            estimate_bytes=None,
        )

    if pre_flight.fail_code is not None:
        # State stays LIVE — do NOT store the stash.
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
            warnings=shape_change_warnings,
        )
        payload = render_preview_response(preview_stash, pre_flight_fail=pre_flight.fail_code)
        # None measurement fields (the check_error sentinel shape) pass
        # through as None — PreviewResponse already declares both GB fields
        # ``float | None`` for exactly this case.
        _used = pre_flight.disk_used_bytes
        _cap = pre_flight.disk_cap_bytes
        payload["pre_flight_disk_used_gb"] = _used / (1024**3) if _used is not None else None
        payload["pre_flight_disk_cap_gb"] = _cap / (1024**3) if _cap is not None else None
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
        warnings=shape_change_warnings,
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

    Implements the 5-step atomic ordering:

    1. Acquire migration lock + verify STAGING + verify not consolidating +
       **construct the candidate config and check it against the tier store**
       (``validate_candidate``).  All three confirm branches (pure
       mode-switch, base swap, general trial) share this gate, so a
       candidate that cannot boot, or that contradicts the tiers already on
       disk, is rejected before the first mutation — before any backup,
       marker, ``state="TRIAL"``, or background task.
    2. Write the pre-migration config backup slot (``backup_live_config``).
    3. Write ``state/trial.json`` marker.
    4. ``promote_config(candidate → configs/server.yaml)`` — re-read, re-hash-check,
       re-validate, atomic swap.
    5. Set ``_state["migration"]["state"] = "TRIAL"``; kick off trial
       consolidation via ``asyncio.create_task``.

    Each step's failure rolls back all previously-completed steps and returns
    an appropriate 5xx error.  The migration lock is unconditionally released
    in a ``finally`` block.

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
    409 ``candidate_invalid_config``
        The candidate parses but cannot be constructed into a bootable config.
        STAGING is preserved; nothing on disk changed.
    409 ``candidate_changed``
        The candidate file changed on disk after it was staged.  Re-preview.
    500 ``backup_write_failed``
        Step 2 failed; no state change.
    500 ``marker_write_failed``
        Step 3 failed; Step 2 backups deleted.
    500 ``config_swap_failed``
        Step 4 failed; marker and backups deleted.
    409 ``disk_pressure``
        Base-swap branch only — the backup store is at its cap, so the
        rollback anchor cannot be written.  Checked before any mutation;
        nothing staged, STAGING retained.
    409 ``store_quarantined``
        The memory store is quarantined (:func:`_store_quarantine_verdict`)
        — a trial reads the live store (the base-swap branch trains
        against it directly). Checked before any mutation.
    """
    from fastapi import HTTPException

    from paramem.server.migration import (
        CandidateChanged,
        CandidateConfigInvalid,
        TrialStash,
        _build_mode_switch_block,
        backup_live_config,
        initial_migration_state,
        promote_config,
        validate_candidate,
    )

    # --- Step 0: memory store must be publishable before a trial reads it ---
    _quarantine_verdict = _store_quarantine_verdict()
    if _quarantine_verdict is not None:
        error, message = refusal_for(
            _quarantine_verdict, doing="starting a migration trial", then="confirm"
        )
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

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
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
    )
    if config is not None:
        state_dir = data_state_dir(config.paths.data).resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = data_state_dir(default_data_dir()).resolve()
        backups_root = (default_data_dir() / "backups").resolve()

    trial_adapter_dir = str((state_dir / "trial" / "adapters").resolve())
    trial_graph_dir = str((state_dir / "trial" / "graph").resolve())

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
        # Reject confirm if a base-swap orchestration is actively running.
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

        # --- Construct the candidate before ANY mutation ---
        # Shared by all three branches below (pure mode-switch, base swap,
        # general trial): each of them renames the candidate over the live
        # config, so each of them must first prove the candidate can boot.
        # A candidate that cannot be constructed would otherwise go live and
        # the next boot would die on it.  The constructed config is discarded —
        # it carries interpolated secrets; the live config is re-loaded from disk.
        try:
            validate_candidate(migration.get("candidate_bytes", b""), live_config_path)
        except CandidateConfigInvalid as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "candidate_invalid_config", "message": str(exc)},
            ) from exc

        # Steps 2–4 are wrapped in try/finally so the lock is always released
        # even on partial failure.
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

            # Back up the live config BEFORE the swap — symmetric with the
            # general-trial path.  This branch drops straight back to LIVE (no
            # trial marker), so the backup slot is the only restore point for a
            # mode switch the operator wants to undo.
            try:
                pre_trial_hash, ms_config_slot = backup_live_config(
                    live_config_path, backups_root, config.security.backups
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "backup_write_failed",
                        "message": f"Failed to write pre-migration config backup: {exc}",
                    },
                ) from exc

            try:
                promote_config(
                    Path(candidate_path_str),
                    live_config_path,
                    expected_sha256=candidate_hash,
                )
            except CandidateChanged as exc:
                try:
                    shutil.rmtree(ms_config_slot)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=409,
                    detail={"error": "candidate_changed", "message": str(exc)},
                ) from exc
            except CandidateConfigInvalid as exc:
                try:
                    shutil.rmtree(ms_config_slot)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=409,
                    detail={"error": "candidate_invalid_config", "message": str(exc)},
                ) from exc
            except Exception as exc:
                try:
                    shutil.rmtree(ms_config_slot)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "config_swap_failed",
                        "message": f"Atomic config rename failed: {exc}",
                    },
                ) from exc

            # Refresh _state["config"] to the new mode and arm the per-tier
            # rebuild (lifespan-mirror; no model reload), then return to LIVE.
            # The candidate was constructed twice before the rename (in-lock gate
            # and inside promote_config), so this load cannot raise on config
            # content — the live file is known-constructible.
            _refresh_config_from_disk_into_state()
            _state["migration"] = initial_migration_state()

            return ConfirmResponse(
                state="LIVE",
                trial_started_at=now_iso,
                pre_trial_config_sha256=pre_trial_hash,
                candidate_config_sha256=candidate_hash,
                backup_paths={"config": str(ms_config_slot.resolve())},
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

            # Guard the base_swap_task slot BEFORE mutating any migration
            # state below: the pre-checks earlier in this handler
            # (current_state != "TRIAL", migration_lock held for this whole
            # confirm) mean no other base-swap orchestration can be running
            # here, but a non-None handle would mean a just-completed
            # orchestration's done-callback has not yet cleared it — refuse
            # rather than silently overwrite a handle
            # _run_boot_completion_tasks/shutdown still expect to find.
            if _state.get("base_swap_task") is not None:
                logger.warning(
                    "/migration/confirm: base_swap_task slot already occupied — "
                    "refusing to launch a second base-swap orchestration"
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "base_swap_active",
                        "message": (
                            "A base-swap orchestration task handle is still present. "
                            "Retry once it clears."
                        ),
                    },
                )

            # Disk-cap door — checked BEFORE any state mutation.  The
            # rollback anchor (write_bundle, inside the orchestration) is
            # cap-gated, so a store at its cap would otherwise refuse the
            # anchor after TRIAL was already committed, wedging the process
            # until the orchestration's own failure classifier runs. Refuse
            # here instead: nothing staged, state unchanged, prune and retry.
            try:
                enforce_disk_cap(backups_root, config.security.backups)
            except DiskCapExceeded as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "disk_pressure", "message": str(exc)},
                ) from exc

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

            # Kick off Phase A as a background task.  Stored in the same slot
            # _run_boot_completion_tasks awaits and shutdown cancels — an
            # unstored asyncio.create_task(...) result is a GC hazard (the
            # event loop only holds a weak reference).
            _state["base_swap_task"] = asyncio.create_task(
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
            _state["base_swap_task"].add_done_callback(
                functools.partial(_clear_state_task, "base_swap_task")
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
        written_slots: list[Path] = []
        try:
            pre_trial_hash, config_slot = backup_live_config(
                live_config_path, backups_root, config.security.backups
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
        # Capture config artifact filename so the rollback handler can resolve
        # the exact A-config file without directory listing.
        # Filter uses endswith(".meta.json") to exclude all sidecar variants
        # (e.g. "config-<ts>.meta.json") regardless of prefix — the old
        # exact-match "meta.json" filter missed prefixed sidecars and caused
        # rollback to restore the sidecar JSON instead of the config artifact.
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
            # Step 3 failure: delete Step 2 backups.
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

        # --- Step 4: Promote the candidate over the live config ---
        # promote_config re-reads the candidate from disk, re-checks its hash
        # against the staged one, re-constructs it at the live path, and only
        # then renames.  Any rejection here leaves the filesystem untouched.
        candidate_path = Path(candidate_path_str)

        def _undo_steps_2_and_3() -> None:
            """Delete the trial marker and every backup slot written above."""
            try:
                clear_trial_marker(state_dir)
            except OSError:
                pass
            for slot in written_slots:
                try:
                    shutil.rmtree(slot)
                except OSError:
                    pass

        try:
            promote_config(candidate_path, live_config_path, expected_sha256=candidate_hash)
        except CandidateChanged as exc:
            _undo_steps_2_and_3()
            raise HTTPException(
                status_code=409,
                detail={"error": "candidate_changed", "message": str(exc)},
            ) from exc
        except CandidateConfigInvalid as exc:
            _undo_steps_2_and_3()
            raise HTTPException(
                status_code=409,
                detail={"error": "candidate_invalid_config", "message": str(exc)},
            ) from exc
        except Exception as exc:
            # Step 4 failure: delete marker and all backups.
            _undo_steps_2_and_3()
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
    state/trial/adapters/), and calls ``_run_extraction_phase`` with
    ``mark_sessions=False``.

    Passing ``mark_sessions=False`` ensures that
    ``session_buffer.mark_consolidated`` is **never** called from the trial
    loop — the transcript sweeper blocks archive+delete on pending sessions.
    Pending sessions remain in the buffer after the trial cycle completes so that
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
            Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
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
        # Read trial_adapter_dir directly from state; do NOT resolve via find_live_slot.
        trial_adapter_dir_str = trial_data.get("trial_adapter_dir", "")
        trial_graph_dir_str = trial_data.get("trial_graph_dir", "")

        # CRITICAL: do NOT override trial_config.paths.data here.
        # Previously this was set to trial_adapter_dir.parent.parent (= data/ha), causing
        # _save_registry to resolve to the LIVE registry path and per-tier
        # key_metadata.json writes to land under the LIVE adapter tree.
        # Isolation is now handled entirely inside _build_trial_loop: the
        # legacy combined registry via loop.trial_registry_path, and
        # per-tier key_metadata.json by construction — loop.output_dir IS
        # trial_adapter_dir, so the fold's per-tier writes (write_tier_slot /
        # publish_tier_registry, via run_consolidation_cycle) land there.

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
        # (new ERROR lines in the trial log).  The `with` closes
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

                speaker_store = _state.get("speaker_store")

                # Closure dict used by _run() to pass graph stash results back to
                # the outer scope without changing _run()'s return type.
                # Keys populated inside the gpu_lock_sync block (where GPU is held):
                #   "pre_trial_graph_path"  — Path (simulate) or None
                #   "pre_trial_graph"       — nx.MultiDiGraph (train) or None
                #   "trial_graph_path"      — Path (simulate) or None
                #   "trial_graph"           — nx.MultiDiGraph (train) or None
                _graph_stash: dict = {}

                def _run():
                    # Temporarily override _state for the trial context so that
                    # _run_extraction_phase reads the trial config and the live
                    # session_buffer, then restore _state after the call.
                    # Trial keeps its own loop (separate adapter/graph dirs).
                    prior_config = _state.get("config")
                    prior_speaker_store = _state.get("speaker_store")
                    _state["config"] = trial_config
                    # speaker_store is passed implicitly through _state.
                    _state["speaker_store"] = speaker_store
                    try:
                        with gpu_lock_sync():
                            # Pre-trial graph capture (train mode only).
                            # In simulate mode the canonical episodic/graph.json on
                            # disk is the pre-trial artifact; migration_status reads it
                            # directly from the production loop's output_dir (unchanged).
                            # In train mode there is no canonical file, so we reconstruct
                            # the graph from adapter weights BEFORE the trial extraction
                            # mutates the adapter state.  This call needs the GPU lock —
                            # reconstruct_graph calls model.generate(); do NOT move it
                            # outside this block or into the async GET handler.
                            _trial_mode = trial_config.consolidation.mode
                            if _trial_mode != "simulate":
                                from paramem.graph.reconstruct import reconstruct_graph

                                _prod_loop = _state.get("consolidation_loop")
                                if _prod_loop is not None:
                                    try:
                                        _pre_result = reconstruct_graph(
                                            _prod_loop,
                                            tier="episodic",
                                            strict=False,
                                        )
                                        _graph_stash["pre_trial_graph"] = _pre_result.graph
                                    except Exception as _rg_exc:  # noqa: BLE001
                                        logger.warning(
                                            "trial: pre_trial graph reconstruct failed"
                                            " (non-fatal, report row will show —): %s",
                                            _rg_exc,
                                        )

                            # Trial path: mark_sessions=False so sessions stay
                            # pending and /migration/rollback can restore queue.
                            _extraction_result = _run_extraction_phase(
                                loop,
                                mark_sessions=False,
                            )

                            # Trial graph capture (after extraction completes).
                            # simulate: stash the newest interim family's BOUND
                            # slot graph.json the trial just wrote (the fold
                            # writes into episodic/interim_<stamp>/<ts2>/graph.json —
                            # resolved via _resolve_bound_graph_path, never a
                            # bare family-root path, which nothing writes).
                            # train: reconstruct the trial loop's adapter weights.
                            if _trial_mode == "simulate":
                                from paramem.memory.interim_adapter import iter_interim_dirs

                                _loop_out = getattr(loop, "output_dir", None)
                                if _loop_out is not None:
                                    _slots = list(iter_interim_dirs(Path(_loop_out)))
                                    if _slots:
                                        # Newest family last (iter_interim_dirs sorts by stamp).
                                        _newest_path = _resolve_bound_graph_path(_slots[-1][1])
                                        _graph_stash["trial_graph_path"] = _newest_path
                            else:
                                from paramem.graph.reconstruct import reconstruct_graph

                                try:
                                    _trial_result = reconstruct_graph(
                                        loop,
                                        tier="episodic",
                                        strict=False,
                                    )
                                    _graph_stash["trial_graph"] = _trial_result.graph
                                except Exception as _rg_exc2:  # noqa: BLE001
                                    logger.warning(
                                        "trial: trial graph reconstruct failed"
                                        " (non-fatal, report row will show —): %s",
                                        _rg_exc2,
                                    )

                            return _extraction_result
                    finally:
                        _state["config"] = prior_config
                        _state["speaker_store"] = prior_speaker_store

                try:
                    summary = await asyncio.get_running_loop().run_in_executor(None, _run)
                except Exception as _exc:  # noqa: BLE001
                    exc_captured = _exc

            # --- Gate evaluation ---
            # live_adapter_dir comes from the PRE-TRIAL config (not the candidate).
            live_config = _state.get("config")
            if live_config is None:
                raise RuntimeError(
                    "trial consolidation: _state['config'] is missing — "
                    "cannot resolve live adapter dir for gate 4"
                )
            live_adapter_dir: Path = live_config.adapter_dir
            trial_adapter_dir = (
                Path(trial_adapter_dir_str)
                if trial_adapter_dir_str
                else data_state_dir(default_data_dir()) / "trial" / "adapters"
            )

            # model IS loop.model — the base model's object identity is fixed
            # at load time, so every holder (including the trial loop)
            # holds the identical PeftModel; no pick is needed.
            results = evaluate_gates(
                model=model,
                tokenizer=tokenizer,
                trial_adapter_dir=trial_adapter_dir,
                live_adapter_dir=live_adapter_dir,
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

        # Stash graph shape artifacts captured inside _run() (under the GPU lock)
        # so migration_status can pass them to build_comparison_report.
        # simulate → Path to the newest interim-slot graph.json (trial) and the
        #            canonical episodic/graph.json (pre_trial, read by migration_status
        #            directly from the production loop's output_dir — no stash needed).
        # train   → in-memory nx.MultiDiGraph from reconstruct_graph (both sides).
        # _stash_trial_graph writes "trial_graph_path" or "trial_graph" to the stash;
        # migration_status reads both and passes whichever is set to build_comparison_report.
        if not session_buffer_empty and "loop" in locals() and loop is not None:
            await _stash_trial_graph(
                trial_graph_path=_graph_stash.get("trial_graph_path"),
                trial_graph=_graph_stash.get("trial_graph"),
                pre_trial_graph=_graph_stash.get("pre_trial_graph"),
            )

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
       ``write_bundle`` again.  Immediately after the fresh-start write, the
       just-written manifest is read back and checked for a tier marked
       weightless (see **Failure semantics** below) — the swap refuses to
       proceed on a rollback anchor that does not capture a fully bound
       store.
    2. **Phase A** — arm the active-store ``train→simulate`` migration state
       file and submit it to a fresh ``BackgroundTrainer`` worker (holds GPU
       lock during execution).  Reconstructs keyed facts from live Mistral
       weights into per-tier ``graph.json`` files, then deletes the adapter
       weight slots.  Marker → ``phaseA_done``.  Config atomically renamed to
       the candidate (Qwen3) variant.
    3. **In-process reload** — two sub-cases based on ``resume_phase``:

       - **Fresh start** (``resume_phase == ""``): drain and reload the base
         in-process via :func:`_gpu_release_internal` (called directly, NOT
         the ``/gpu/release`` route handler — this orchestration legitimately
         holds ``base_swap_active=True`` and the route's own guard would
         refuse it) followed by :func:`_apply_config_live`.
         ``_gpu_release_internal`` drains BOTH the base model AND the voice
         pipeline to cloud-only; ``_apply_config_live`` then re-reads the
         renamed-on-disk config and loads it on the clean GPU, so the VRAM
         gate sees accurate free bytes and the reload fits in-process
         without a restart.
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
    5. **Recall gate** — Phase B's ``migrate()`` call probes every entry in
       the tier (uncapped by construction — there is no sampling knob to
       set).  If any tier fails the 1.0 gate, the
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
       ``migrate()`` on all_tiers_done), clear the trial marker, reset the
       migration stash to ``state="LIVE"`` carrying a terminal ``gates``
       report via :func:`_finish_base_swap` (status ``pass``), then
       best-effort resolve any active ``migration_phase_failed`` incident
       left by a prior failed attempt at this same swap.  That resolve runs
       in its own try/except **after** the "pass" report is already
       published: its failure (e.g. a corrupt ``incidents.json``) is logged
       and swallowed, never allowed to re-enter this coroutine's own
       exception handler and repaint a genuinely completed swap as failed.

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

    **In-flight guard**

    ``_state["migration"]["base_swap_active"]`` is ``True`` while this
    coroutine is actively executing phases.  It is cleared in ``finally`` so it
    is ``False`` whenever the coroutine has exited — whether by success,
    failure, or deferred return.  ``POST /migration/confirm`` and
    ``POST /migration/rollback`` reject with 409 while the flag is ``True``.
    Rollback remains available when the flag is ``False`` and a ``phaseA_done``
    or ``phaseB`` marker exists (stranded deferred swap).

    **Failure semantics**

    - Setup failure (``write_trial_marker`` never committed — tracked
      directly via ``_marker_written``, never inferred from the marker being
      absent on disk, since the success path also clears the marker before
      it publishes): gates → ``setup_failed`` via :func:`_finish_base_swap`,
      which resets the migration stash to ``state="LIVE"``.  The live config
      was never renamed, so ``POST /migration/preview`` is immediately
      retryable.  ``write_bundle`` may still have committed before a later
      pre-marker step raised (tracked via ``_bundle_written``) — when it did,
      the retention-immune bundle slot is surfaced as ``gates["bundle_path"]``
      rather than silently claimed away; it is never auto-deleted.  A resume
      launched at ``phaseA_done``/``phaseB`` starts with both flags already
      ``True`` (a resume's own precondition is that its marker still reads
      back), so a mid-resume failure is never misclassified as setup_failed —
      it falls through to the ``phase_b_failed`` case below, ``state`` stays
      ``TRIAL``.  A fresh-start setup failure also covers the rollback-anchor
      gate: the rollback anchor must capture a fully bound store, so a
      capture that had to mark any tier's adapter record
      ``weightless_cause`` (``write_bundle`` found an on-disk payload that
      looked trained but no slot matched the live registry) refuses the swap
      here, before any mutation — ``RuntimeError`` names the torn tier(s),
      cause(s), and the remedy (repair the store — retire the affected keys
      via the erase endpoint or restore a healthy backup — then retry the
      swap) and is classified ``setup_failed`` by this same arm.  The
      bundle is real and preserved (surfaced via ``gates["bundle_path"]``
      exactly as any other setup-failure bundle) — it is the truthful
      forensic capture of the broken tier, not something to hide.  This gate
      only runs on the fresh-start path: a resume never re-captures, and its
      marker's mere presence already proves the original capture passed it.
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
    from paramem.server.migration import promote_config

    # ── In-flight guard: set base_swap_active so confirm/rollback can reject ──
    migration = _state.get("migration")
    if isinstance(migration, dict):
        migration["base_swap_active"] = True

    # Durable-mutation tracking, set directly at the point each write commits
    # (never inferred from disk state afterward — see the classifier below).
    # A resume launch already has both from a prior fresh-start run (the
    # "_resume_marker is None" guard a few lines down raises if that is not
    # true), so both start True for resume_phase in ("phaseA_done", "phaseB").
    _bundle_written = resume_phase not in ("", "phaseA")
    _marker_written = resume_phase not in ("", "phaseA")

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
            # Fresh start — build per-tier adapter_dirs dict from the live config.
            adapter_dirs: dict[str, Path] = {
                _tier_name: Path(config.adapter_dir) / _tier_name
                for _tier_name in config.tier_config_map()
            }

            data_dir = Path(config.paths.data)
            speaker_profiles_path = data_dir / "speaker_profiles.json"

            # ── Step 1: Bundle backup (rollback anchor) ───────────────────────
            # Written exactly once — before any mutation.  Resume paths never
            # reach this block (guarded by the resume_phase check above).
            bundle_slot = write_bundle(
                config_path=live_config_path,
                adapter_dirs=adapter_dirs,
                backups_root=backups_root,
                backups_cfg=config.security.backups,
                meta_fields={"tier": "pre_base_swap", "label": f"pre_base_swap_{new_model}"},
                adapter_scope="live",
                speaker_profiles_path=(
                    speaker_profiles_path if speaker_profiles_path.exists() else None
                ),
                candidate_config_path=Path(candidate_path_str),
            )
            bundle_slot_str = str(bundle_slot.resolve())
            _bundle_written = True

            # ── Rollback-anchor validity gate ──────────────────────────────
            # A rollback replays this bundle unconditionally (no defensive
            # handling in the rollback branch of POST /migration/rollback)
            # because the bundle is guaranteed to be a working-state capture.
            # write_bundle marks a tier's adapter record weightless_cause
            # ("torn_train_slot") when the tier's on-disk payload looked
            # trained but no slot matched the live registry — that tier was
            # already broken before this swap touched anything. Refuse to
            # BEGIN rather than anchor a rollback to a store that cannot be
            # restored to a working state.  No mutation has happened yet:
            # _marker_written is still False, so raising here is classified
            # setup_failed by the except-arm below — the live config stays
            # untouched and the swap is immediately retryable once the store
            # is repaired.  The bundle itself is left on disk (retention-immune,
            # never auto-deleted) and surfaced via gates["bundle_path"] as a
            # truthful forensic capture of the broken state.
            _bundle_manifest = read_bundle_manifest(bundle_slot)
            _weightless_tiers = {
                _name: _record["weightless_cause"]
                for _name, _record in _bundle_manifest.adapters.items()
                if _record["weightless_cause"] is not None
            }
            if _weightless_tiers:
                _tier_report = "; ".join(
                    f"{_name} ({_cause})" for _name, _cause in sorted(_weightless_tiers.items())
                )
                raise RuntimeError(
                    "base-swap refused before any mutation: rollback anchor "
                    f"{bundle_slot_str} captured {_tier_report} without adapter "
                    "weights — repair the store first (retire the affected keys "
                    "via the erase endpoint or restore a healthy backup), then "
                    "retry the swap."
                )

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
            _marker_written = True

            # Arm the train→simulate active-store migration state file.
            migration_state = MigrationState.for_mode_switch(
                source_mode="train", target_mode="simulate"
            )
            save_state(Path(config.adapter_dir), migration_state)

            loop = get_or_create_consolidation_loop(_state)

            bt = _build_bg_trainer(config)
            _state["background_trainer"] = bt
            _state["consolidation_loop"]._bg_trainer = bt

            # asyncio.Event lets us await the BG-worker result from this coroutine.
            # Capture the running event loop at creation time so the worker thread
            # can signal completion via call_soon_threadsafe even when
            # _state["event_loop"] is not yet populated (e.g. during unit tests).
            _phase_a_aio_loop = asyncio.get_event_loop()
            done_event = asyncio.Event()
            phase_a_error: list[BaseException] = []

            def _run_phase_a_on_worker() -> None:
                """Run on the BG-trainer worker thread under the GPU lock.

                The whole body runs inside try/finally: ``BackgroundTrainer.
                _run_callable_queue`` catches and merely logs any exception
                escaping the submitted job — it never re-raises to this
                coroutine — so ``done_event.set()`` is the ONLY signal that
                can wake ``await done_event.wait()`` below.  ANY exception
                here — including one newly reachable from ``migrate()``'s
                "0 registered tiers but on-disk content exists" guard (now
                also tripped by a strict-load refusal on a foreign-shaped
                tier registry) — must still reach the ``finally`` or the
                await hangs forever instead of surfacing the failure.
                ``except BaseException`` (not just ``Exception``) mirrors
                ``BackgroundTrainer.submit_and_wait``'s own wrapper — a
                worker-thread ``BaseException`` must still set the event
                rather than escape ``_run_callable_queue``'s narrower
                ``except Exception`` and kill the persistent worker thread.
                Caught exceptions are appended to ``phase_a_error`` (checked
                after the await) rather than swallowed.
                """
                from paramem.server.active_store_migration import load_state as _phase_a_load_state

                try:
                    _fresh_state = _phase_a_load_state(Path(config.adapter_dir))
                    if _fresh_state is None:
                        phase_a_error.append(RuntimeError("Phase A: migration state file vanished"))
                        return
                    updated = migrate(loop, config, _fresh_state)
                    # migrate() re-creates and retrains each tier it touches
                    # directly on loop.model (create_adapter + train), not
                    # through the fold's go-live promote path — re-snapshot
                    # here even on a partial run, since a tier already
                    # migrated before a later tier's failure keeps its new
                    # weight state regardless of the overall outcome.
                    _record_tier_weight_state(_state, loop.model, config)
                    if not updated.all_tiers_done(loop.store.tiers_with_registry()):
                        first_fail = next(iter(updated.failed_tiers.values()), "unknown")
                        phase_a_error.append(RuntimeError(f"Phase A incomplete: {first_fail}"))
                except BaseException as exc:  # noqa: BLE001 — must surface, never hang the awaiter
                    phase_a_error.append(exc)
                finally:
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
            promote_config(
                Path(candidate_path_str),
                live_config_path,
                expected_sha256=candidate_hash,
            )

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
                #   2. _gpu_release_internal: the only path that actually reclaims
                #      the old base + voice here (the bare reload-release left it
                #      resident at 1.34 GiB free; release drains to ~6.5 GiB
                #      free).  Calls the internal directly (not the `/gpu/release`
                #      route handler) because this orchestration legitimately
                #      holds `base_swap_active=True` — the route's first guard
                #      exists to refuse *external* callers during an active swap
                #      and would 409 on our own in-flight flag.
                #   3. _apply_config_live: re-reads the renamed-on-disk config and
                #      reloads the NEW base with a full rebuild (gpu_acquire would
                #      keep the stale in-memory config).  _live_reload_base_model
                #      recomputes the VRAM topology for the new model so the gate
                #      uses its footprint, not the old base's.
                # base_swap_active is True, so neither re-enters this orchestration.
                loop = None
                bt = None
                _release_result = await _gpu_release_internal()
                # _gpu_release_internal's success shape is the dict below; a
                # refusal (503 consolidating, in-lock TOCTOU recheck — see its
                # docstring) is a JSONResponse instead. A non-HTTP caller like
                # this orchestration must not silently treat that refusal as
                # success and press on into _apply_config_live against a base
                # model that was never actually released — raise so the
                # existing outer exception handler records this as
                # phase_a_failed (marker is still at "phaseA" here), which is
                # the correct outcome for a refused release.
                if not isinstance(_release_result, dict):
                    _release_detail = getattr(_release_result, "body", b"").decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(
                        "base-swap fresh-start reload: _gpu_release_internal refused the "
                        f"release (status={getattr(_release_result, 'status_code', '?')}): "
                        f"{_release_detail}"
                    )
                _apply_result = await asyncio.get_running_loop().run_in_executor(
                    None, _apply_config_live
                )

                # Proceed to Phase B iff a reload ran and landed — the only
                # shape where the new model is actually resident.  _gpu_release_
                # internal above already released the GPU, so every other
                # shape leaves mode cloud-only; this predicate reads the
                # primitive's own return dict instead of re-inferring the
                # outcome from a state re-read.
                if not (_apply_result["applied_live"] and _apply_result["skipped"] is None):
                    completed_at = datetime.now(timezone.utc).isoformat()
                    restart_required_reason = _apply_result.get("restart_required_reason")
                    apply_cloud_only_reason = _apply_result.get("cloud_only_reason")
                    # A reload was attempted (and its outcome recorded in
                    # cloud_only_reason) whenever the apply reached the
                    # reload dispatch — that includes a MIXED R-PORT delta,
                    # which sets restart_required_reason (the carve is still
                    # signalled) but falls through to the reload rather than
                    # short-circuiting before it (see the R-PORT mixed-delta
                    # branch above).  So restart_required_reason alone does
                    # NOT mean "never attempted" — cloud_only_reason (or the
                    # no-op skip shape) is the reliable signal that a reload
                    # ran.
                    attempted = apply_cloud_only_reason is not None or (
                        _apply_result.get("skipped") == "no_change"
                    )
                    if attempted:
                        # A reload was attempted and failed (a handled
                        # failure — see _live_reload_base_model's closed
                        # vocabulary).  _apply_config_live's refresh-before-
                        # release ordering already committed config B to
                        # _state["config"] before the release, so a later
                        # plain /gpu/acquire genuinely reloads the NEW model;
                        # the phaseA_done marker (already written above) is
                        # what makes /gpu/acquire's deferred-resume hook
                        # re-launch Phase B once that reload lands.
                        gates_cloud_only_reason = apply_cloud_only_reason or "reload_deferred"
                        message = (
                            f"Phase A complete but base-model reload deferred "
                            f"(cloud_only_reason={gates_cloud_only_reason!r}). "
                            "Phase B will run automatically once the new model "
                            "is loaded (POST /gpu/acquire triggers this)."
                        )
                    else:
                        # The apply was never attempted (lock_timeout /
                        # consolidating / paths_change / a pure-port-only
                        # carve) — config A is still the in-memory config, so
                        # a plain /gpu/acquire would reload the OLD model and
                        # this same deferral would just repeat.  Config B is
                        # already on disk (the atomic rename in Step 2); only
                        # a restart picks it up.  cloud_only_reason is reserved
                        # for a genuine cloud-only cause and stays None here —
                        # restart_required_reason already names the cause.
                        gates_cloud_only_reason = None
                        message = (
                            f"Phase A complete but the config apply was never attempted "
                            f"(restart_required_reason={restart_required_reason!r}). "
                            "Config B is already on disk — restart the service to pick "
                            "it up and resume Phase B; POST /gpu/acquire will NOT help "
                            "here."
                        )
                    await _update_trial_gates(
                        {
                            "status": "reload_deferred",
                            "completed_at": completed_at,
                            "cloud_only_reason": gates_cloud_only_reason,
                            "restart_required_reason": restart_required_reason,
                            "message": message,
                        }
                    )
                    logger.warning(
                        "base-swap reload deferred: cloud_only_reason=%s "
                        "restart_required_reason=%s; Phase B not started",
                        gates_cloud_only_reason,
                        restart_required_reason,
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
        # Unlike _run_active_store_migration_sync (which reuses the singleton via
        # _active_bg_trainer), Phase B runs on a freshly reloaded base model so
        # _build_bg_trainer is correct here.
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

        loop_b = get_or_create_consolidation_loop(_state)

        bt_b = _build_bg_trainer(config_b)
        _state["background_trainer"] = bt_b
        _state["consolidation_loop"]._bg_trainer = bt_b

        _phase_b_aio_loop = asyncio.get_event_loop()
        done_event_b = asyncio.Event()
        phase_b_error: list[BaseException] = []

        def _run_phase_b_on_worker() -> None:
            """Run Phase B on the BG-trainer worker thread under the GPU lock.

            The whole body runs inside try/finally: ``BackgroundTrainer.
            _run_callable_queue`` catches and merely logs any exception
            escaping the submitted job — it never re-raises to this
            coroutine — so ``done_event_b.set()`` is the ONLY signal that
            can wake ``await done_event_b.wait()`` below.  ANY exception
            here — including one newly reachable from
            ``load_registries_from_disk``'s per-tier ``KeyRegistry.load``
            (a foreign-shaped on-disk registry now raises instead of
            silently loading empty) or the sibling ``migrate()`` call —
            must still reach the ``finally`` or the await hangs forever
            instead of surfacing the failure.  ``except BaseException``
            (not just ``Exception``) mirrors ``BackgroundTrainer.
            submit_and_wait``'s own wrapper — a worker-thread
            ``BaseException`` must still set the event rather than escape
            ``_run_callable_queue``'s narrower ``except Exception`` and
            kill the persistent worker thread.  Caught exceptions are
            appended to ``phase_b_error`` (checked after the await) rather
            than swallowed.
            """
            from paramem.server.active_store_migration import load_state as _phase_b_load_state
            from paramem.server.consolidation import (
                load_max_tier_cycle as _phase_b_load_max_tier_cycle,
            )

            try:
                _fresh_state_b = _phase_b_load_state(Path(config_b.adapter_dir))
                if _fresh_state_b is None:
                    phase_b_error.append(
                        RuntimeError("Phase B: migration state file vanished before Phase B ran")
                    )
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
                # Bookkeeping + promotion state — mirrors the ordinary lifespan-boot
                # hydration order (registries, then bookkeeping). commit_tier_slot's
                # bookkeeping write is per-tier now (each tier writes only its own
                # rows, never truncating another tier's file), but seed_key_metadata
                # still needs the store's bookkeeping already loaded — its
                # promoted_keys rebuild reads the per-key promoted flag off the
                # store. load_bookkeeping_from_disk is the sole boot loader for
                # per-key bookkeeping (speaker_id, relation_type,
                # reinforcement_count, ...); create_consolidation_loop's
                # construction-time seed_key_metadata call
                # (paramem/server/consolidation.py) ran against the still-empty
                # base-swap store and found no bookkeeping to derive promoted_keys
                # from, so it is re-run here now that the registries above are
                # loaded — loop_b picks up the correct promoted_keys/cycle_count
                # the same way the ordinary live-singleton mode-switch venue
                # already does (its store is fully hydrated at boot, so this is a
                # no-op change for it).
                loop_b.store.load_bookkeeping_from_disk(config_b.adapter_dir)
                _phase_b_cycle_count = _phase_b_load_max_tier_cycle(config_b.adapter_dir)
                if _phase_b_cycle_count is not None:
                    loop_b.seed_key_metadata(_phase_b_cycle_count)
                updated_b = migrate(loop_b, config_b, _fresh_state_b)
                # loop_b.model IS _state["model"] here (loop_b was built by
                # get_or_create_consolidation_loop off the live _state,
                # post-Step-3 base reload) — re-snapshot even on a partial
                # run; a later Step 6 reload re-derives it again once the
                # swap fully lands.
                _record_tier_weight_state(_state, loop_b.model, config_b)
                if not updated_b.all_tiers_done(loop_b.store.tiers_with_registry()):
                    first_fail = next(iter(updated_b.failed_tiers.values()), "unknown")
                    phase_b_error.append(RuntimeError(f"Phase B incomplete: {first_fail}"))
            except BaseException as exc:  # noqa: BLE001 — must surface, never hang the awaiter
                phase_b_error.append(exc)
            finally:
                _phase_b_aio_loop.call_soon_threadsafe(done_event_b.set)

        bt_b.submit(_run_phase_b_on_worker, inference_fallback_adapter="episodic")
        await done_event_b.wait()

        if phase_b_error:
            raise phase_b_error[0]

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
        # as the Phase A → Phase B reload at Step 3).  /gpu/release and
        # /gpu/acquire cannot race us here — both doors refuse with 409
        # ``base_swap_active`` for as long as this coroutine is executing
        # (the guards at the top of ``gpu_release`` and ``gpu_acquire``),
        # not merely because this flag happens to still read True.
        #
        # Voice drain/restore is owned by _live_reload_base_model: the
        # primitive drains STT/TTS to CPU before its VRAM gate (preventing
        # the ~4.3 GiB voice footprint from blocking the gate) and restores
        # to GPU after a successful partial reload.  ``async with gpu_lock()``
        # holds the non-reentrant lock across the dispatch — the same
        # discipline the plain-acquire and auto-reclaim reload callers use —
        # so ``lock_held=True`` is passed to keep the internal
        # ``_set_voice_pipeline_profile`` calls from re-acquiring it.
        #
        # If the reload fails internally it leaves the server cloud-only with
        # cloud_only_reason set; the swap is already complete on disk, so we
        # still mark status=pass and let the next /gpu/acquire recover.
        loop_b = None
        bt_b = None
        try:
            from paramem.server.gpu_lock import gpu_lock

            _loop = asyncio.get_running_loop()
            async with gpu_lock():
                _reload_reason = await _loop.run_in_executor(
                    None, lambda: _live_reload_base_model(lock_held=True)
                )
            if _reload_reason is not None:
                logger.warning(
                    "base-swap: post-Phase-B live reload failed (%s); weights are on "
                    "disk but the server stays cloud-only until /gpu/acquire or restart",
                    _reload_reason,
                )
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

        completed_at = datetime.now(timezone.utc).isoformat()
        # Logged BEFORE the publish (not after) so nothing between here and
        # the end of the try can raise once "pass" is live — see the
        # try/except immediately below for the one statement that remains
        # after the publish.
        logger.info(
            "base-swap orchestration complete: old=%s new=%s",
            old_model,
            new_model,
        )
        await _finish_base_swap(
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
        # Best-effort housekeeping against external on-disk state
        # (incidents.json), run AFTER "pass" is already published.  This is
        # boundary error handling for that external state, not suppression
        # of an orchestration failure: a genuinely completed swap must never
        # be repainted as failed by a later, unrelated I/O error — so this
        # is caught and logged rather than left to re-enter the except-arm's
        # classifier below (which would overwrite the "pass" gates and mint
        # a false migration_phase_failed incident).  A prior failed attempt
        # for this same swap may have left an active migration_phase_failed
        # incident (setup_failed / phase_a_failed / phase_b_failed); clear it
        # now that the retry succeeded, so a permanently red attention row
        # does not survive a clean run.
        try:
            resolve_incidents_by_type(
                data_state_dir(_state["config"].paths.data),
                "migration_phase_failed",
                reason=f"base-swap completed: {old_model} → {new_model}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "base-swap orchestration: resolve_incidents_by_type failed after a "
                "successful swap (old=%s new=%s) — the swap itself completed; prior "
                "migration_phase_failed incidents remain unresolved in /status until "
                "manually resolved or the next successful swap",
                old_model,
                new_model,
            )

    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc).isoformat()
        # Determine which phase failed.  ``_marker_written`` is the tracked
        # invariant (set at the point ``write_trial_marker`` actually
        # committed, never inferred from disk afterward) — the marker being
        # absent on disk is NOT proof it was never written: the success path
        # clears it (clear_trial_marker) BEFORE publishing via
        # _finish_base_swap, so even a failure at that publish step itself
        # would otherwise look identical, on disk, to "nothing was ever
        # written".  Once "pass" actually publishes, nothing after it in the
        # try can reach this except-arm (resolve_incidents_by_type has its
        # own narrow try/except precisely so a housekeeping failure there
        # can never repaint a completed swap as failed).  The marker is
        # still read here because a *present* marker's ``base_swap_phase``
        # is the only way to tell phase_a_failed from phase_b_failed.
        _failed_marker = None
        _marker_read_failed = False
        try:
            _failed_marker = read_trial_marker(state_dir)
        except Exception:  # noqa: BLE001
            _marker_read_failed = True

        if _failed_marker is None and not _marker_read_failed and not _marker_written:
            _phase, _status = "setup", "setup_failed"
        elif _failed_marker is not None and _failed_marker.base_swap_phase in ("phaseA", ""):
            _phase, _status = _failed_marker.base_swap_phase, "phase_a_failed"
        else:
            _phase = (
                getattr(_failed_marker, "base_swap_phase", "unknown")
                if _failed_marker is not None
                else "unknown"
            )
            _status = "phase_b_failed"

        if _status == "setup_failed":
            # The marker was never written, so promote_config never ran —
            # the live config is untouched and there is no marker to roll
            # back.  A bundle slot MAY still exist (write_bundle can commit
            # before a later step in the same fresh-start attempt raises,
            # e.g. the pre_trial_hash read or write_trial_marker itself);
            # when it does, it is a real, retention-immune (30-day) rollback
            # anchor, not orphaned residue — surfaced via ``bundle_path`` so
            # the CLI reports it honestly instead of claiming nothing
            # changed.  Never auto-deleted here; the operator prunes it
            # deliberately if unwanted.
            _setup_failed_gates = {
                "status": _status,
                "exception": str(exc),
                "completed_at": completed_at,
            }
            if _bundle_written:
                _setup_failed_gates["bundle_path"] = bundle_slot_str
            await _finish_base_swap(_setup_failed_gates)
        else:
            await _update_trial_gates(
                {"status": _status, "exception": str(exc), "completed_at": completed_at}
            )
        logger.exception("base-swap orchestration failed (phase=%s): %s", _phase, exc)
        # Record the phase failure as a durable incident.  The existing
        # bundle + POST /migration/rollback path is unchanged; this adds
        # durability + /status visibility alongside the trial-gates marker.
        # Recorded after the gates/state update above — record_incident
        # writes to data_state_dir(config.paths.data), which the migration reset
        # does not touch.
        record_incident(
            data_state_dir(_state["config"].paths.data),
            type="migration_phase_failed",
            key=_status,
            severity="failed",
            summary=f"Base-swap migration {_status}",
            detail={"phase": _phase, "exception": str(exc), "at": completed_at},
        )
        # Bundle and marker are preserved for rollback.  setup_failed has no
        # marker to preserve (never written) and preserves the bundle only
        # when write_bundle already committed (see ``bundle_path`` above).

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

    Decision table (overall status rollup):

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

    Registry isolation: ``loop.output_dir`` is set to ``trial_adapter_dir``
    (below), so every per-tier file the fold's commit primitives write —
    ``indexed_key_registry.json`` and ``key_metadata.json`` alike, via
    ``write_tier_slot`` / ``publish_tier_registry`` (the trial consolidation
    run) or ``commit_tier_slot`` (``commit_main_tiers``'s copy-forward of
    the unchanged main adapters) — lands inside the trial adapter tree.
    The loop carries no separate metadata-path override; there is nothing
    left to isolate beyond ``output_dir`` itself.

    The previous pattern (``trial_config.paths.data = trial_adapter_dir.parent.parent``)
    is removed: it pointed ``paths.data`` back to ``data/ha`` and caused both
    registry writers to resolve to the LIVE paths.  The adapter output path is
    now set via ``loop.output_dir`` only, leaving ``trial_config.paths.data``
    alone so the paths resolved from configuration (sessions, debug, prompts)
    remain valid.

    Args:
        trial_adapter_dir: Required.  A ``None`` value would leave
            ``loop.output_dir`` on ``create_consolidation_loop``'s production
            default (``config.adapter_dir``) — and since the fold state dir
            is derived from ``output_dir.parent`` via
            :func:`~paramem.training.stage_ledger.data_state_dir`, an
            un-overridden trial would read, overwrite, and dispose
            PRODUCTION fold state.  Refused loudly instead; there is no
            fallback.

    Raises:
        ValueError: When *trial_adapter_dir* is ``None``, or when a tier's
            ``indexed_key_registry.json`` under *trial_adapter_dir* exists
            but is not KeyRegistry-shaped — propagated from
            :meth:`~paramem.memory.store.MemoryStore.load_registries_from_disk`
            (batch, all-or-nothing per
            :meth:`~paramem.memory.store.MemoryStore.read_registries_from_disk`).
            A malformed tier registry aborts the trial loudly rather than
            degrading to an empty store; the sole caller
            (``_run_trial_consolidation``) catches this and records
            ``status="trial_exception"``.
    """
    if trial_adapter_dir is None:
        raise ValueError(
            "_build_trial_loop: trial_adapter_dir is required — None would leave "
            "loop.output_dir on its production default (config.adapter_dir), so "
            "the trial would read, overwrite, and dispose PRODUCTION fold state"
        )

    from paramem.memory.store import MemoryStore as _MemoryStore
    from paramem.server.consolidation import create_consolidation_loop

    # Trial path: construct a fresh, isolated store that mirrors the trial
    # adapter dir's registries.  Do NOT reuse the live ``_state["memory_store"]``
    # — the trial must not pollute the production store.  A registry load
    # failure propagates rather than degrading to an empty store — see
    # Raises below.
    trial_store = _MemoryStore()
    trial_store.load_registries_from_disk(trial_adapter_dir)
    loop = create_consolidation_loop(model, tokenizer, trial_config, trial_store)

    loop.output_dir = trial_adapter_dir
    trial_adapter_dir.mkdir(parents=True, exist_ok=True)

    # Donor stores stay on the LIVE root: the trial's adapters are cold by
    # construction, so without this every trial pays a full inline donor
    # build into its own scratch tree. Borrowing is read-only, so a trial
    # can neither add to nor prune the live donor stores.
    loop.borrow_donor_cache(trial_config.adapter_dir)

    # Redirect the legacy combined-SimHash registry write to a trial-isolated
    # directory under the trial root (sibling of adapters/, e.g.
    # data/ha/state/trial/trial_registry/).
    trial_registry_dir = trial_adapter_dir.parent / "trial_registry"
    loop.trial_registry_path = trial_registry_dir / "registry.json"

    return loop


async def _finish_base_swap(gates: dict) -> None:
    """Return the migration stash to LIVE carrying the swap's terminal report.

    Holds ``migration_lock`` for the whole reset — same discipline as
    :func:`_update_trial_gates` (and stricter than the reset it replaces on
    the success path, which ran unlocked).  ``recovery_required`` and the
    trial stash's identity fields (``started_at`` / ``candidate_config_sha256``
    / ``backup_paths``) are carried forward from the stash being replaced, so
    ``GET /migration/status`` still reports WHICH swap finished and where its
    bundle is; ``gates`` is replaced by *gates*.  Every other field returns to
    its LIVE sentinel, so preview/confirm are available again and
    accept/rollback see a non-``TRIAL`` state (refuse) rather than a stale
    ``TRIAL``.  The next ``POST /migration/preview`` clears the report — it
    rebuilds the stash with ``trial=None``.

    Callers: the base-swap orchestration's success arm (``gates={"status":
    "pass", ...}``) and its ``setup_failed`` classifier arm (the trial marker
    was never committed, so the live config was never renamed and the
    process must not stay wedged in TRIAL with no marker to roll back;
    ``gates`` may carry a ``bundle_path`` when ``write_bundle`` already
    committed before the failure).

    Deadlock note: ``_run_base_swap_orchestration`` never holds
    ``migration_lock`` itself — ``POST /migration/confirm`` creates the
    orchestration task inside its own ``async with lock`` block but returns
    without an intervening ``await``, so the task's first run starts after
    that lock has released.  The boot-completion resume launch
    (``app.py`` lifespan, around the ``_base_swap_resume_marker`` handling)
    and the ``/gpu/acquire`` deferred-resume relaunch both call this same
    coroutine from outside any lock, so acquiring it here is safe.
    """
    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    async with lock:
        from paramem.server.migration import TrialStash, initial_migration_state

        prior = _state.get("migration") or {}
        prior_trial = prior.get("trial") or {}
        prior_recovery = list(prior.get("recovery_required") or [])

        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery
        _state["migration"]["trial"] = TrialStash(
            started_at=prior_trial.get("started_at", ""),
            pre_trial_config_sha256="",
            candidate_config_sha256=prior_trial.get("candidate_config_sha256", ""),
            backup_paths=prior_trial.get("backup_paths", {}),
            trial_adapter_dir="",
            trial_graph_dir="",
            gates=gates,
        )


async def _update_trial_gates(gates: dict) -> None:
    """Update ``_state["migration"]["trial"]["gates"]`` under ``migration_lock``.

    The trial coroutine ``_run_trial_consolidation`` has ``await`` points
    (run_in_executor for both build_trial_loop and the consolidation executor)
    before this function runs, so a concurrent ``/migration/cancel`` may execute
    and clear ``_state["migration"]["trial"]`` between the trial coroutine
    starting and this update. ``/migration/cancel`` holds ``migration_lock``
    while clearing trial state, so this writer must hold the same lock to
    observe a consistent view.

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


def _resolve_bound_graph_path(tier_root: "Path") -> "Path | None":
    """Resolve *tier_root*'s BOUND slot's ``graph.json`` for the migration
    comparison report.

    Nothing writes a tier-root ``graph.json`` any more — the payload lives in
    a timestamped slot under *tier_root*, written via
    :func:`~paramem.adapters.slot.write_slot`. Composes the same two
    primitives every other bound-slot reader in the package does
    (:func:`~paramem.adapters.manifest.tier_registry_sha256` +
    :func:`~paramem.adapters.manifest.find_live_slot` —
    :mod:`paramem.memory.source`, :mod:`paramem.server.active_store_migration`,
    :mod:`paramem.backup.backup`) rather than the heavier registry↔slot
    binding oracle, which is for boot verification and publish gating, not a
    read-only display value.

    Best-effort, matching this pair's own documented convention (see
    ``tier_registry_sha256``'s docstring: "callers at a boot boundary that
    must degrade rather than fail ... catch locally there") — a read/decrypt
    failure on the registry, or any failure resolving the slot, degrades to
    ``None`` rather than raising, because a comparison report must never
    crash the request; the caller passes ``None`` straight through to
    ``_summarise_graph``, which renders ``"—"``.

    Args:
        tier_root: Directory holding the tier's ``indexed_key_registry.json``
            at its root — a main tier root or an interim family root.

    Returns:
        Path to the bound slot's ``graph.json``, or ``None`` when no slot is
        bound or resolution failed.
    """
    import pyrage

    from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
    from paramem.adapters.slot import payload_filename

    try:
        live_hash = tier_registry_sha256(tier_root)
        slot = find_live_slot(tier_root, live_hash)
    except (OSError, RuntimeError, pyrage.DecryptError) as exc:
        logger.warning(
            "migration comparison report: could not resolve bound slot for %s: %s",
            tier_root,
            exc,
        )
        return None
    if slot is None:
        return None
    return slot / payload_filename("simulate")


async def _stash_trial_graph(
    *,
    trial_graph_path: "Path | None",
    trial_graph: "object | None",
    pre_trial_graph: "object | None",
) -> None:
    """Store graph shape artifacts on the trial stash for the comparison report.

    Called from ``_run_trial_consolidation`` after the fold completes so that
    ``migration_status`` can pass them to ``build_comparison_report`` without
    reading the in-memory merger graph (cleared at cycle-end by the finally block).

    Mirrors ``_update_trial_gates`` in structure; holds ``migration_lock``
    to avoid a race with ``/migration/cancel`` that could clear the trial stash
    between the fold completing and this write.

    When the trial stash has already been cleared (e.g. concurrent rollback),
    this is a silent no-op — the same safe behaviour as ``_update_trial_gates``.

    Parameters
    ----------
    trial_graph_path:
        ``Path`` to the newest interim-slot ``graph.json`` written by the
        simulate fold.  ``None`` when the fold ran in train mode or produced
        no interim slots.
    trial_graph:
        In-memory ``nx.MultiDiGraph`` reconstructed from the trial adapter
        weights (train mode).  ``None`` in simulate mode or on reconstruct
        failure.
    pre_trial_graph:
        In-memory ``nx.MultiDiGraph`` reconstructed from the production
        adapter weights before the trial ran (train mode).  ``None`` in
        simulate mode (the production ``episodic/graph.json`` is used instead,
        resolved directly by ``migration_status`` from the production loop's
        ``output_dir``).
    """
    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    async with lock:
        migration = _state.get("migration")
        if migration is None:
            return
        trial = migration.get("trial")
        if trial is None:
            return
        # Store whatever was captured; migration_status reads each key
        # independently and falls back to None (→ "—" in the report row).
        trial["trial_graph_path"] = trial_graph_path
        trial["trial_graph"] = trial_graph
        trial["pre_trial_graph"] = pre_trial_graph


# Accept-eligible gate statuses (set membership for forward-compat).
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
    # top-level status (see _ACCEPT_ELIGIBLE_STATUSES).
    comparison_report: dict | None = None
    if (
        ms == "TRIAL"
        and gates.get("status") in _ACCEPT_ELIGIBLE_STATUSES
        and gates.get("completed_at")
    ):
        # Resolve graph shape from stashed artifacts.
        #
        # Pre-trial simulate: the production episodic tier's BOUND slot
        # graph.json (nothing writes a tier-root graph.json any more —
        # resolved via _resolve_bound_graph_path, on disk before the trial
        # ran, no GPU access needed).
        # Pre-trial train: in-memory graph reconstructed before extraction ran
        # (captured proactively under the GPU lock in _run_trial_consolidation).
        #
        # Trial simulate: Path to the newest interim family's bound slot
        # graph.json the trial wrote, stashed by _stash_trial_graph.
        # Trial train: in-memory graph reconstructed from trial adapter weights,
        # stashed by _stash_trial_graph.
        #
        # _summarise_graph returns "—" gracefully for any None/absent value.
        pre_trial_graph_path: Path | None = None
        pre_trial_graph: object | None = trial.get("pre_trial_graph")
        if pre_trial_graph is None:
            # simulate mode: resolve the production episodic tier's bound slot.
            _loop_obj = _state.get("consolidation_loop")
            if _loop_obj is not None:
                _out_dir = getattr(_loop_obj, "output_dir", None)
                if _out_dir is not None:
                    from paramem.memory.interim_adapter import adapter_slot_root_for_name

                    pre_trial_graph_path = _resolve_bound_graph_path(
                        adapter_slot_root_for_name(Path(_out_dir), "episodic")
                    )

        trial_graph_path: Path | None = trial.get("trial_graph_path")
        trial_graph: object | None = trial.get("trial_graph")

        comparison_report = build_comparison_report(
            gates=gates,
            pre_trial_graph_path=pre_trial_graph_path,
            trial_graph_path=trial_graph_path,
            pre_trial_graph=pre_trial_graph,
            trial_graph=trial_graph,
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

    # pre_flight_fail is always None here by construction, not by omission:
    # STAGING is entered ONLY from /migration/preview's success branch
    # (the sole `state="STAGING"` construction site, above), which only runs
    # after pre_flight.fail_code is None; migration_recovery.py resumes a
    # torn migration to TRIAL, never STAGING. So every reachable STAGING
    # state already passed pre-flight — there is no failed check to report.
    payload = render_preview_response(migration, pre_flight_fail=None)
    return MigrationDiffResponse(**payload)


_RESTART_HINT: str = "systemctl --user restart paramem-server"


@app.post("/migration/accept", response_model=AcceptResponse, dependencies=[Depends(require_admin)])
async def migration_accept():
    """Promote trial config B to live, archive the trial adapter, and clear trial state.

    Only valid when the server is in TRIAL state and gates have finished with an
    accept-eligible status (``pass`` or ``no_new_sessions``).

    5-step atomic ordering (marker cleared before adapter/graph move — rationale below):

    1. Re-verify preconditions inside lock (state, gates).
    2. Build rotation slot for trial adapter archive.
    3. **Clear trial marker** (BEFORE adapter/graph move).
    4. Move trial adapter + graph into the rotation slot.
    5. Refresh drift state and set restart banner.

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
    409 ``store_quarantined``
        The memory store is quarantined (:func:`_store_quarantine_verdict`).
        Checked before any mutation.
    500 ``trial_archive_failed``
        Could not create the rotation slot for the trial adapter.
    """
    from fastapi import HTTPException

    from paramem.server.drift import ConfigDriftState, compute_config_hash
    from paramem.server.migration import initial_migration_state

    # --- Store must be publishable before promoting a trial live ---
    _quarantine_verdict = _store_quarantine_verdict()
    if _quarantine_verdict is not None:
        error, message = refusal_for(
            _quarantine_verdict, doing="accepting a migration trial", then="accept"
        )
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

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
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
    )
    if config is not None:
        state_dir = data_state_dir(config.paths.data).resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = data_state_dir(default_data_dir()).resolve()
        backups_root = (default_data_dir() / "backups").resolve()

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

        # --- Step 3: Clear trial marker BEFORE adapter/graph move ---
        # Rationale: if marker-clear fails, nothing else has mutated yet.
        # If rotation below fails after marker-clear, state/trial/adapters/ is still
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

        # --- Step 4: Move trial adapter into the rotation slot; delete trial graph dir ---
        # Non-fatal: config + marker are already coherent. Rotation is cosmetic.
        # The trial graph path is stashed in _state["migration"]["trial"]["trial_graph_path"]
        # by _stash_trial_graph; the file (episodic/interim_<stamp>/graph.json in simulate
        # mode; a reconstructed in-memory graph in train mode) lives inside
        # trial_adapter_dir which is moved/deleted below.
        # The trial_graph dir (if it exists) is deleted unconditionally as
        # cleanup; it is empty by design.
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
        # Caller ordering precondition: _apply_config_live compares disk_hash
        # against config_drift["loaded_hash"] (hash of config A, captured at boot).
        # The drift refresh below must happen AFTER _apply_config_live so the
        # no-op skip (disk_hash == loaded_hash) does not fire on the accept path.
        #
        # Apply ordering:
        #  a) Set synchronous maintenance guard BEFORE dispatching executor:
        #     mode="cloud-only" so the scheduler's mode != "local" defer fires
        #     during the brief window between migration-state reset and apply.
        #  b) Dispatch _apply_config_live via run_in_executor (runs synchronously
        #     from the accept handler's perspective — we await the result).
        #  c) Refresh config_drift AFTER a successful apply (loaded_hash now B).
        #  d) Build the banner and response.  The server does NOT fire _restart_service
        #     for R-PORT; the CLI prompts the operator and runs the restart_hint command.

        # Synchronous maintenance guard + dispatch + restore-if-untouched
        # — see _apply_config_live_guarded for the full rationale.
        apply_result = await _apply_config_live_guarded()

        applied_live: bool = apply_result.get("applied_live", False)
        apply_reason: str | None = apply_result.get("restart_required_reason")
        restart_eligible: bool = apply_result.get("restart_eligible", False)
        apply_cloud_only_reason: str | None = apply_result.get("cloud_only_reason")

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
            restart_eligible=restart_eligible,
            cloud_only_reason=apply_cloud_only_reason,
        )


@app.post("/migration/rollback", dependencies=[Depends(require_admin)])
async def migration_rollback():
    """Restore config A from backup, archive trial adapter, clear trial state.

    Valid from TRIAL at any time (no gate-status check required).

    8-step atomic ordering (marker cleared before rotation):

    1. Re-verify inside lock (state=TRIAL).
    2. Snapshot B into rollback_pre_mortem backup.
    3. Resolve A config artifact from marker; decrypt it and **construct it as if
       it already sat at the live config path, and check it against the tier
       store** (``validate_candidate``) — a backup is a config that was
       validated against a schema (and a store) that may since have grown new
       guards, so restoring it is a second door onto the same "unbootable /
       store-contradicting config goes live" defect a candidate promotion
       closes.
    4. Atomic rename A artifact → live config path.
    5. **Clear trial marker** (BEFORE rotation).
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
    400 ``backup_unbootable``
        Step 3 construction failed; live config untouched; pre-mortem backup deleted.
    500 ``config_restore_failed``
        Step 4 rename failed; pre-mortem backup deleted.
    500 ``marker_clear_failed``
        Step 5 marker clear failed; internal inconsistency.
    """
    from fastapi import HTTPException

    from paramem.backup.types import ArtifactKind
    from paramem.server.migration import (
        CandidateConfigInvalid,
        initial_migration_state,
        validate_candidate,
    )
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
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
    )
    if config is not None:
        state_dir = data_state_dir(config.paths.data).resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = data_state_dir(default_data_dir()).resolve()
        backups_root = (default_data_dir() / "backups").resolve()

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

        # --- In-flight guard — reject rollback while orchestration runs ---
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
        # Store-side convergence: this branch enters quarantine deliberately
        # BEFORE restore_bundle rewrites the tree (the same enter-quarantine
        # -> restore-tree -> lift sequence POST /backup/restore uses — see
        # _enter_store_quarantine's docstring for the shared marker/incident
        # shape). The lift itself is not a separate call here: the base
        # MODEL is also changing, so the full release+reload below
        # (_apply_config_live_guarded(force=True) -> _live_reload_base_model
        # -> _build_runtime_components) already re-runs the store step as
        # part of its normal full-rebuild, which is what clears the
        # quarantine on success. A restore_bundle failure leaves the
        # quarantine in place with the failure recorded as its cause.
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

            # Deliberate quarantine entry BEFORE the tree rewrite — the store
            # goes offline for the duration of the rewrite; chat keeps
            # serving on the HA/cloud paths exactly as any quarantine does.
            #
            # LOAD-BEARING INVARIANT: this entry, the tree rewrite below, and
            # the ledger dispose further down (`_sl.dispose(state_dir)`) run
            # in one synchronous stretch with NO `await` in between -- see
            # the identical invariant note at `POST /backup/restore`'s own
            # deliberate quarantine entry.  A deliberate quarantine that left
            # a pending ledger observable across an await boundary here could
            # race the arbitrator's resume-pending-first arm (which now
            # treats quarantined-AND-pending as "resume it") against a tree
            # this rollback is still mid-rewriting.
            _enter_store_quarantine(
                config,
                reason=f"rolling back a base-swap migration (bundle {bundle_slot_path.name})",
            )

            try:
                _restore_bundle_fn(
                    bundle_slot_path,
                    data_dir=data_dir_rb,
                    config_path=live_config_path,
                    restore_config=True,
                )
            except (BundleManifestError, FingerprintMismatchError) as exc:
                # Failed restore -- leave the quarantine in place, updated
                # with the actual failure as its cause.
                _enter_store_quarantine(config, exc)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "bundle_corrupt",
                        "message": f"Base-swap rollback bundle is corrupt: {exc}",
                    },
                ) from exc
            except RuntimeError as exc:
                _enter_store_quarantine(config, exc)
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

            # This restore rewrites every tier wholesale, which a pending
            # consolidation event's ledger would otherwise classify FOREIGN
            # on its next resume.  Path-only -- no live ConsolidationLoop
            # required, so this fires in every server mode, cloud-only
            # included.
            from paramem.training import stage_ledger as _sl

            _sl.dispose(state_dir)
            # Every exit of a pending record resolves the incident naming it --
            # this rollback's discard is one of the wholesale-tier-rewrite
            # sites, the same invariant as the other dispose call sites.
            resolve_incidents_by_type(state_dir, "consolidation_resume_blocked")

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
            # shared synchronous maintenance guard.  force=True makes the
            # reload run UNCONDITIONALLY — no config-hash sentinel: the
            # quarantine entered above must not be left stranded by a no-op
            # skip, and _apply_config_live's own skip logic stays intact for
            # every other caller (see its *force* parameter doc).  The
            # dispatched reload's own full component rebuild is what runs the
            # store-side lift and clears the quarantine on success — no
            # separate lift call is needed here.  This path does not consume
            # the apply result — a reload failure leaves the quarantine (and
            # cloud-only mode) in place; the operator recovers via
            # GET /integrity and a repeat POST /migration/rollback or
            # POST /gpu/acquire.
            await _apply_config_live_guarded(force=True)

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
                backups_root=backups_root,
                backups_cfg=None,  # undo anchor — exempt from the disk cap
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
        # Read marker from disk to get config_artifact_filename.
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
        # yaml.safe_load to fail on the next server start.
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

        # --- Step 3b: Construct A as if it already sat at the live config path ---
        # A backup is a config that was validated against a POSSIBLY OLDER schema.
        # New load-time guards (e.g. max_interim_count=0 + mode=simulate) can reject
        # bytes that were bootable when the backup was written. The pre-mortem
        # backup (step 2) already exists, so refusing here costs nothing — the
        # operator can still recover the artifact manually if they need it.
        try:
            validate_candidate(plaintext_bytes, live_config_path)
        except CandidateConfigInvalid as exc:
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "backup_unbootable",
                    "message": (
                        f"A-config backup cannot be constructed into a bootable config: {exc}. "
                        "The live config was NOT modified."
                    ),
                },
            ) from exc

        # Initialize before the try block so the except cleanup guard never
        # raises UnboundLocalError if the assignment itself (e.g. datetime.now)
        # raises before pending_restore is set.
        pending_restore: Path | None = None
        try:
            _ts_suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            pending_restore = live_config_path.parent / f".pending-rollback-{_ts_suffix}.yaml"
            # Write the temp file at 0o600 to prevent a plaintext-exposure
            # window under the default umask (0644).  os.O_EXCL ensures the file
            # is created atomically; write via fd.
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
            # Clean up pending temp if it was created.
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

        # --- Step 5: Clear trial marker BEFORE rotation ---
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
        # Rollback restored disk to A and config_drift["loaded_hash"] is still A
        # (rollback does NOT refresh drift — see comment at step 6240).
        # So _apply_config_live will take the no-op skip (disk_hash == loaded_hash)
        # and return applied_live=True, skipped="no_change" without GPU churn.
        #
        # Synchronous maintenance guard + dispatch + restore-if-untouched
        # — rollback's apply normally takes the no-op skip
        # (disk==memory==A), returning with the guard untouched, so the restore
        # inside _apply_config_live_guarded keeps the server from sticking
        # cloud-only.
        apply_result = await _apply_config_live_guarded()

        applied_live: bool = apply_result.get("applied_live", False)
        apply_reason: str | None = apply_result.get("restart_required_reason")
        restart_eligible: bool = apply_result.get("restart_eligible", False)
        apply_cloud_only_reason: str | None = apply_result.get("cloud_only_reason")

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
                "restart_eligible": restart_eligible,
                "cloud_only_reason": apply_cloud_only_reason,
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
            restart_eligible=restart_eligible,
            cloud_only_reason=apply_cloud_only_reason,
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
        Artifact kind string (``"config"`` | ``"graph"`` | ``"snapshot"`` |
        ``"resume"`` | ``"snapshot_bundle"``).
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
    incompatible:
        ``True`` when this is a ``snapshot_bundle`` slot written at a
        ``bundle_schema_version`` this build no longer understands — the
        slot is enumerated (visible) but ``POST /backup/restore`` refuses
        it. Always ``False`` for a per-artifact record or a bundle at the
        current version.
    """

    backup_id: str
    kind: str
    tier: str
    timestamp: str
    size_bytes: int
    label: str | None
    path: str
    incompatible: bool = False


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
        ``["snapshot_bundle"]`` (self-contained recovery bundle) — the one
        comprehensive, restorable artifact.  ``"config"`` and ``"graph"``
        remain independently selectable extras.
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
        For ``snapshot_bundle`` restores this maps each restored adapter name
        to its new slot directory, plus a ``"<adapter_name>_key_metadata"``
        entry per adapter whose ``key_metadata.json`` was restored, and
        (optionally) ``"speaker_profiles"`` and ``"config"``.
    backed_up_pre_restore:
        Mapping of kind → safety backup slot path taken before restore.
        For ``snapshot_bundle`` restores the key is ``"bundle"`` and the value
        is the pre-restore safety bundle slot path (or ``""`` when skipped
        because the live store was empty).
    restored_adapters:
        List of adapter names restored from the bundle.  Empty for
        ``config``-kind restores.
    pruned_orphans:
        Orphan adapters removed by the clean-slate sweep during restore: whole
        main tiers and interim families that were on disk but absent from the
        bundle's recovery set.  Each entry carries
        ``{"name": <adapter_name>, "kind": "interim"|"main", "active_keys": <int>}``.
        Routine within-tier stale-slot cleanup is logged but not listed here.
        Empty when no orphan adapters were pruned.
    serving:
        ``True`` when the server is fully serving with the restored artifacts
        live — no restart, no residual quarantine — by the time this response
        is returned.  ``False`` means either an operator restart is still
        required to converge the restore (a ``config``-kind restore, or a
        ``snapshot_bundle`` restore with ``restore_config=True`` — both leave
        their existing restart posture unchanged, since a same-base lift
        would be wrong when the config may have changed the base model), or
        the post-restore lift itself re-quarantined the store (see
        ``quarantine_cause``).
    quarantine_cause:
        The current ``_state["store_quarantine"]`` cause dict when
        ``serving`` is ``False`` because the store is quarantined
        (``snapshot_bundle`` restores only — a plain ``config``-kind restore
        never quarantines, since it touches no adapter tree); ``None`` when
        the store is not quarantined.
    """

    restored: dict[str, str]
    backed_up_pre_restore: dict[str, str]
    restored_adapters: list[str] = []
    pruned_orphans: list[dict] = []
    serving: bool
    quarantine_cause: dict | None = None


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
    ``"graph"`` | ``"snapshot"`` | ``"resume"`` | ``"snapshot_bundle"``).
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
            backups_root = (default_data_dir() / "backups").resolve()
    else:
        backups_root = (default_data_dir() / "backups").resolve()

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
                incompatible=record.incompatible,
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


def _create_backup(
    kinds: list[str] | None,
    tier: str,
    label: str | None,
) -> BackupCreateResponse:
    """Validate a backup request, run the scheduled-backup pipeline, and persist the result.

    The callable core shared by ``POST /backup/create`` (the route supplies
    ``kinds``/``tier``/``label`` straight from ``BackupCreateRequest`` — see
    that route's docstring, ``tier`` defaulting to ``"manual"``) and the
    boot-completion catch-up task (:func:`_run_boot_completion_tasks`, which
    calls this off the event loop via ``asyncio.to_thread`` with
    ``kinds=list(config.security.backups.artifacts)`` — the same artifact
    list the ``paramem-backup`` systemd timer's standalone runner delegates
    with, see ``paramem/backup/__main__.py``, ``tier="daily"`` — the tier
    that same runner always posts with, and ``label=None``).

    Delegates to ``run_scheduled_backup`` with a per-call shallow-cloned
    config (``.security.backups.artifacts`` replaced by *kinds*, defaulting
    to ``["snapshot_bundle"]`` when *kinds* is ``None``/empty).
    ``snapshot_bundle`` produces a single self-contained bundle slot under
    ``backups_root/snapshot/`` containing the full recovery set (config,
    registry, adapter weights, speaker profiles); the server holds the
    ``PARAMEM_DAILY_PASSPHRASE`` needed to decrypt registries for per-tier
    hash resolution, which is why scheduled backups are server-mediated.
    ``"config"`` and ``"graph"`` remain independently selectable extras.

    Persists the result via ``update_backup_state`` so the next ``/status``
    reflects the freshly-updated ``last_success_at``.

    Reads the live config, the consolidation loop, and the config path from
    module state (``_state``) at call time.

    Args:
        kinds: Artifact kinds to back up. ``None``/``[]`` -> default
            ``["snapshot_bundle"]``.
        tier: Retention tier the slot is filed under.
        label: Optional operator-supplied annotation.

    Returns:
        The ``BackupCreateResponse`` describing what was written/skipped.

    Raises:
        HTTPException: 400 ``kind_invalid`` / ``tier_invalid`` on bad input
            (surfaced as an HTTP response by the route; the boot-completion
            task never triggers this branch since it always calls with a
            valid tier and the operator's own configured artifact list).
    """
    import dataclasses

    from fastapi import HTTPException

    from paramem.backup.runner import run_scheduled_backup
    from paramem.backup.state import update_backup_state

    _VALID_KINDS = {"config", "graph", "snapshot_bundle"}

    # Validate kinds.
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
    if tier not in _VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tier_invalid",
                "message": f"tier must be one of {sorted(_VALID_TIERS)}; got {tier!r}",
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

    state_dir = data_state_dir(config.paths.data).resolve()
    backups_root = (config.paths.data / "backups").resolve()
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
    )
    loop = _state.get("consolidation_loop")

    result = run_scheduled_backup(
        server_config=proxy_config,
        loop=loop,
        state_dir=state_dir,
        backups_root=backups_root,
        live_config_path=live_config_path,
        tier=tier,
        label=label,
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
    "/backup/create",
    response_model=BackupCreateResponse,
    dependencies=[Depends(require_admin)],
)
async def backup_create(req: BackupCreateRequest):
    """Take an immediate backup of the requested artifacts.

    Delegates to :func:`_create_backup` with the request's ``kinds``,
    ``tier`` (default ``"manual"``), and ``label``.  The scheduled systemd
    timer posts here with ``tier="daily"`` so the self-contained recovery
    bundle is captured under daily retention; the boot-completion catch-up
    task calls :func:`_create_backup` directly (in-process, off the event
    loop) instead of looping back through this HTTP route.

    Errors
    ------
    400 ``kind_invalid``
        When any entry in ``kinds`` is not a recognised artifact kind.
    400 ``tier_invalid``
        When ``tier`` is not a recognised retention tier.
    """
    return _create_backup(req.kinds, req.tier, req.label)


@app.post(
    "/backup/restore",
    response_model=BackupRestoreResponse,
    dependencies=[Depends(require_admin)],
)
async def backup_restore(req: BackupRestoreRequest):
    """Restore a backup atop the live store — the RECOVERY door: it stays
    open in cloud-only mode and under an existing store quarantine.

    Supports two restore kinds:

    - ``kind="config"`` — restore a single ``server.yaml`` from a per-artifact
      config slot.  Unchanged mechanism (no live-apply dispatch, no store
      quarantine — a config-kind restore touches no adapter tree): an
      operator restart is still what converges it, reported via
      ``serving=False`` in the response (see ``BackupRestoreResponse``).
    - ``kind="snapshot_bundle"`` — restore a complete self-contained recovery
      set (adapter weights, per-tier registries, speaker profiles, and
      optionally config) from a bundle slot.

    Sequence for ``snapshot_bundle`` (decrypt-probe, then safety bundle, then
    the atomic tree rewrite, registry written LAST, then convergence):

    1. Verify preconditions (no STAGING, and every arm
       :func:`_consolidation_dispatch_guards` checks with
       ``include_cloud_only=False`` — see the guard comment in the body). A
       pending consolidation event record does NOT refuse here; see the
       per-kind handling in step 4.
    2. Locate the slot by ``backup_id``.
    3. Dispatch to the appropriate kind handler.
    4. For ``snapshot_bundle``:
       a. Verify all file hashes in ``bundle.meta.json`` against on-disk bytes.
       b. Decrypt-probe encrypted metadata files (BEFORE any mutation).
       c. Enter store quarantine deliberately (:func:`_enter_store_quarantine`)
          — the store goes offline for the duration of the rewrite; chat
          keeps serving on the HA/cloud paths exactly as any quarantine does.
       d. Take a manual safety bundle of the current live state.
       e. Atomic restore via ``restore_bundle()``, registry written LAST. A
          failure here leaves quarantine in place with the failure as its
          cause.
       f. Converge: when ``restore_config`` restored a config (the base
          model may have changed), a same-base lift would be wrong — leave
          the quarantine as entered and let an operator restart converge it,
          same restart posture as the ``config`` kind above. When the bundle
          records weightless-marked tiers (``RestoreResult.
          weightless_adapters`` — registry keys with no weight slot), a lift
          would publish silently-unrecallable keys, so the quarantine stays
          with a cause naming the marked tiers and the repair doors. Otherwise
          (same base by construction, no markings) re-mount adapters from the
          restored slots onto the resident model (no-op in cloud-only mode)
          and lift the store (:func:`_lift_quarantined_store`) — no restart
          needed.

    Errors
    ------
    409 ``staging_active``
        Migration state is STAGING (no shared-guard equivalent).
    409 ``trial_active`` | ``consolidating`` | ``training_active`` |
    ``base_swap_active``
        One of :func:`_consolidation_dispatch_guards`'s five busy arms is
        set (:func:`refusal_for` maps the verdict to the error body). A
        pending consolidation event record does NOT refuse here (see step 4
        above). ``training_active`` matters here specifically because
        between the tree rewrite and the on-demand re-mount a running
        background trainer could ``find_live_slot`` on the restored slot
        and stomp it with a checkpoint.
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
    400 ``backup_unbootable``
        (``kind="config"`` only) The decrypted backup cannot be constructed into a
        bootable config.  The safety backup of the current live config was already
        written; the live config itself was NOT modified.
    500 ``config_restore_failed``
        Atomic rename of the restore temp file failed.
    500 ``bundle_restore_failed``
        Unexpected error during bundle restore. The store is left quarantined
        with the failure as its cause.
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
    from paramem.server.migration import (
        CandidateConfigInvalid,
        initial_migration_state,
        validate_candidate,
    )

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
        # No shared-guard equivalent — STAGING is a tier-migration-trial
        # state active_consolidation's vocabulary does not model at all
        # (it is neither a busy arm nor a pending-event arm), so it stays a
        # hand-rolled check here.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "staging_active",
                "state": "STAGING",
                "message": ("Cannot restore during STAGING. Cancel the migration first."),
            },
        )

    # Every other busy arm is the shared five-arm vocabulary
    # (_consolidation_dispatch_guards): consolidating, background training,
    # an active migration TRIAL, and an active base-swap migration —
    # composed with include_cloud_only=False because restore is a RECOVERY
    # door and must stay open in cloud-only mode (a cloud-only server is
    # exactly a state a restore may need to run in). A pending consolidation
    # event record (a stage ledger on disk with no fold actively running)
    # does NOT refuse here: for kind="snapshot_bundle" the ledger is
    # disposed immediately after the bundle restore (the tier rewrite would
    # otherwise leave it classifying everything FOREIGN on the next resume
    # — see the dispose call below); for kind="config" it is left untouched,
    # since a config restore rewrites no tier. Quarantine is likewise NOT
    # composed here (_store_quarantine_verdict is deliberately excluded) —
    # restore must remain open under an EXISTING quarantine; a deliberate
    # re-entry below just updates the cause (see _enter_store_quarantine).
    verdict = _consolidation_dispatch_guards(include_cloud_only=False)
    if verdict is not None:
        error, message = refusal_for(verdict, doing="restoring a backup", then="restore")
        raise HTTPException(status_code=409, detail={"error": error, "message": message})

    config = _state.get("config")
    if config is not None:
        try:
            backups_root = (config.paths.data / "backups").resolve()
        except (AttributeError, TypeError):
            backups_root = (default_data_dir() / "backups").resolve()
    else:
        backups_root = (default_data_dir() / "backups").resolve()

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
        Path(_state["config_path"]) if _state.get("config_path") else DEFAULT_SERVER_CONFIG_PATH
    )

    # --- Step 3: Dispatch by kind ---

    # -------------------------------------------------------------------------
    # SNAPSHOT_BUNDLE branch
    # -------------------------------------------------------------------------
    if target_record.kind == ArtifactKind.SNAPSHOT_BUNDLE:
        # Derive data_dir from config.paths.data.
        try:
            data_dir = config.paths.data.resolve()
        except (AttributeError, TypeError):
            data_dir = default_data_dir().resolve()

        # Deliberate quarantine entry BEFORE the tree rewrite -- the store
        # goes offline for the duration; chat keeps serving on the HA/cloud
        # paths exactly as any quarantine does.
        #
        # LOAD-BEARING INVARIANT: this entry, the tree rewrite below, and the
        # ledger dispose further down (`_sl.dispose(_restore_state_dir)`) run
        # in one synchronous stretch with NO `await` in between -- the
        # arbitrator's resume-pending-first arm now treats a
        # quarantined-AND-pending dispatch as "resume it"
        # (`_dispatch_consolidation`'s docstring, step 3), so a deliberate
        # quarantine that left a pending ledger observable across an await
        # boundary here could race a resume attempt against a tree this
        # restore is still mid-rewriting. Keep the whole
        # enter-quarantine -> rewrite -> dispose sequence await-free.
        _enter_store_quarantine(config, reason=f"restoring backup {req.backup_id}")

        try:
            result = _restore_bundle(
                target_record.slot_dir,
                data_dir=data_dir,
                config_path=live_config_path,
                restore_config=req.restore_config,
            )
        except BundleManifestError as exc:
            _enter_store_quarantine(config, exc)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "bundle_corrupt",
                    "message": f"Bundle manifest invalid or schema-mismatched: {exc}",
                },
            ) from exc
        except FingerprintMismatchError as exc:
            _enter_store_quarantine(config, exc)
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
            _enter_store_quarantine(config, exc)
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
            # restore_bundle raises RestoreAbortedError when the atomic
            # write phase (step 5) fails after the safety bundle was already
            # captured (step 4).  Surface the safety_slot path so the operator
            # can recover without searching server logs.
            _enter_store_quarantine(config, exc)
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
            _enter_store_quarantine(config, exc)
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

        # This restore rewrites every tier wholesale, which a pending
        # consolidation event's ledger would otherwise classify FOREIGN on
        # its next resume.  Path-only -- no live ConsolidationLoop required,
        # so this fires in every server mode, cloud-only included.
        from paramem.training import stage_ledger as _sl

        _restore_state_dir = _sl.data_state_dir(data_dir)
        _sl.dispose(_restore_state_dir)
        # Every exit of a pending record resolves the incident naming it --
        # this restore's discard is one of the wholesale-tier-rewrite sites,
        # the same invariant as the other dispose call sites.
        resolve_incidents_by_type(_restore_state_dir, "consolidation_resume_blocked")

        # Build restored dict: adapter name → new slot path, plus each
        # adapter's own key_metadata.json (per-tier now, not a single global
        # registry path), plus profiles/config.
        restored_map: dict[str, str] = {}
        for adapter_name in result.restored_adapters:
            from paramem.memory.interim_adapter import adapter_slot_root_for_name

            tier_root = adapter_slot_root_for_name(data_dir / "adapters", adapter_name)
            restored_map[adapter_name] = str(tier_root)
            _km_path = tier_root / "key_metadata.json"
            if _km_path.exists():
                restored_map[f"{adapter_name}_key_metadata"] = str(_km_path)
        if (data_dir / "speaker_profiles.json").exists():
            restored_map["speaker_profiles"] = str(data_dir / "speaker_profiles.json")
        if result.restored_config:
            restored_map["config"] = str(live_config_path)

        safety_slot_str = str(result.safety_slot) if result.safety_slot is not None else ""

        if result.pruned_orphans:
            if "migration" not in _state:
                _state["migration"] = initial_migration_state()
            if "recovery_required" not in _state["migration"]:
                _state["migration"]["recovery_required"] = []
            _state["migration"]["recovery_required"].append(
                f"Pruned {len(result.pruned_orphans)} orphan interim adapter "
                f"families during restore: "
                + ", ".join(e["name"] for e in result.pruned_orphans)
                + f" — safety bundle at {safety_slot_str} can restore them if needed."
            )

        # --- Convergence ---
        serving = False

        if result.restored_config:
            # The bundle's config was written to disk and MAY have changed
            # the base model — a same-base lift would be wrong here (see
            # _enter_store_quarantine's cause above). This restore keeps its
            # existing restart posture: quarantine stays exactly as entered
            # (its cause still names this restore), and NO live-apply is
            # dispatched — an operator restart, like the config-kind branch
            # below, is what converges it. Routing this arm through the
            # live-apply machinery instead is a legitimate alternative but
            # was not taken: it would also change the config-kind branch's
            # long-standing restart posture, which is out of scope here.
            logger.info(
                "backup_restore: restore_config=True restored a config that may "
                "have changed the base model — leaving the store quarantined "
                "for an operator restart rather than attempting a same-base lift "
                "(backup_id=%s)",
                req.backup_id,
            )
        elif result.weightless_adapters:
            # The bundle records tiers captured WITHOUT weights (registry
            # keys with no weight slot — the capture-path marking). At the
            # file layer this shape is indistinguishable from a legitimate
            # fresh/simulate tier, so a lift would PUBLISH a store whose
            # marked keys are silently unrecallable. Skip re-mount + lift
            # entirely and keep the store offline: re-enter the quarantine
            # with a cause naming the marked tiers — that same call
            # re-records the store_quarantined incident, which is the
            # durable signal (the marker itself is process state and a
            # restarted boot cannot re-derive this shape from disk).
            _marked = ", ".join(
                f"{tier} ({cause})" for tier, cause in sorted(result.weightless_adapters.items())
            )
            _enter_store_quarantine(
                config,
                reason=(
                    f"restored backup {req.backup_id} with weightless tiers: {_marked} — "
                    "registry keys have no weight slot; stale-mark the affected keys "
                    "via POST /debug/erase-keys, or restore a healthy bundle"
                ),
            )
            logger.warning(
                "backup_restore: bundle %s carries weightless tiers (%s) — "
                "lift skipped, store stays quarantined",
                req.backup_id,
                _marked,
            )
        else:
            # Same base by construction (no config was restored) — the
            # lightweight convergence: release the two process-lifetime
            # holders that pin the PRE-restore model/store (same treatment
            # as _release_base_model_in_process's holders 2 and 3, minus
            # releasing the base model itself), re-mount adapters from the
            # restored slots onto the resident model (no-op in cloud-only
            # mode), then lift the store. The base model's object identity
            # never changes (_remount_adapters_from_disk mutates the SAME
            # PeftModel in place — detach then ensure_resident_tiers, never
            # an unwrap), so this release is about the loop/store wrappers,
            # not the model handle. Without it, _state["consolidation_loop"]
            # keeps its old .store (the pre-restore store
            # _lift_quarantined_store replaces) — the next fold's
            # lazily-cached loop would then run against a stale store.
            # Nulling here means the next fold's get_or_create_consolidation_loop
            # rebuilds fresh against the post-restore _state["memory_store"].
            # Both releases and the remount+lift are GPU-touching when a
            # model is resident, so all of it runs off the event loop under
            # gpu_lock.
            def _restore_converge_sync() -> bool:
                # Re-resolve rather than close over the handler's pre-lock
                # capture — same door-staleness argument as
                # /speaker/forget's _forget_sync.
                _config = _state["config"]
                _bt = _state.get("background_trainer")
                if _bt is not None:
                    try:
                        _bt.release()
                    except Exception:
                        logger.exception(
                            "Error releasing background trainer during restore convergence"
                        )
                _loop = _state.get("consolidation_loop")
                if _loop is not None:
                    try:
                        _loop.release()
                    except Exception:
                        logger.exception(
                            "Error releasing consolidation loop during restore convergence"
                        )
                _state["background_trainer"] = None
                _state["consolidation_loop"] = None
                _remount_adapters_from_disk(_config)
                return _lift_quarantined_store(_config)

            from paramem.server.gpu_lock import gpu_lock

            async with gpu_lock():
                loop_aio = asyncio.get_running_loop()
                lifted = await loop_aio.run_in_executor(None, _restore_converge_sync)
            serving = lifted

            if not serving:
                logger.error(
                    "backup_restore: post-restore lift re-quarantined the store "
                    "(backup_id=%s) — see GET /integrity for the cause",
                    req.backup_id,
                )

        quarantine_cause = None if serving else (_state.get("store_quarantine") or {}).get("cause")

        return BackupRestoreResponse(
            restored=restored_map,
            backed_up_pre_restore={"bundle": safety_slot_str},
            restored_adapters=result.restored_adapters,
            pruned_orphans=result.pruned_orphans,
            serving=serving,
            quarantine_cause=quarantine_cause,
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
            backups_root=backups_root,
            backups_cfg=None,  # undo anchor — exempt from the disk cap
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

    # --- Step 5b: Construct the backup as if it already sat at the live path ---
    # A backup is a config that was validated against a POSSIBLY OLDER schema; new
    # load-time guards can reject bytes that were bootable when the backup was
    # written.  The safety backup above already exists, so refusing here costs
    # nothing — the operator's current live config is unharmed and recoverable.
    try:
        validate_candidate(plaintext_bytes, live_config_path)
    except CandidateConfigInvalid as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "backup_unbootable",
                "message": (
                    f"Backup cannot be constructed into a bootable config: {exc}. "
                    f"The live config was NOT modified. Safety backup at {safety_slot_path}."
                ),
            },
        ) from exc

    # --- Step 6: Atomic restore ---
    restore_pending = live_config_path.with_suffix(live_config_path.suffix + ".restore-pending")
    try:
        # Write at 0o600 to prevent plaintext exposure under the default umask
        # (0644) during the rename window.
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

    # --- Step 7: this branch's existing restart posture ---
    # Unchanged mechanism: a config-kind restore only ever swaps
    # server.yaml on disk — no live-apply dispatch, no adapter tree touched,
    # no store quarantine. An operator restart is what converges it, same as
    # before this change; only the response shape is new (see
    # BackupRestoreResponse's serving field).
    return BackupRestoreResponse(
        restored={"config": str(live_config_path)},
        backed_up_pre_restore={"config": safety_slot_path},
        restored_adapters=[],
        serving=False,
        quarantine_cause=None,
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
            state_dir = data_state_dir(config.paths.data).resolve()
            backups_cfg = config.security.backups
        except (AttributeError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "prune_failed", "message": f"Config error: {exc}"},
            ) from exc
    else:
        from paramem.server.config import ServerBackupsConfig

        backups_root = (default_data_dir() / "backups").resolve()
        state_dir = data_state_dir(default_data_dir()).resolve()
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


def _relay_route(
    text: str,
    history: list[dict] | None,
    config,
    *,
    cloud_permitted: bool,
    ha_client=None,
    cloud_agent=None,
    language: str | None = None,
    speaker_id: str | None = None,
    identity_absent: bool = False,
    model=None,
    tokenizer=None,
) -> ChatResult:
    """Route queries served by the relay path: HA / cloud / local base model, no PA memory.

    Serves two distinct callers through the one leg: (1) server-wide
    cloud-only mode (no local model loaded at all — the original condition
    this function was written for, formerly named ``_cloud_only_route``),
    and (2) a per-request speakerless turn (``ServingPath.RELAY``) served
    this way regardless of server mode (``identity_absent=True``).  In
    server-wide cloud-only mode ``model``/``tokenizer`` are genuinely
    ``None`` (no local model exists — callers pass ``_state["model"]``
    verbatim); in local mode with ``identity_absent=True`` they are the
    live base model/tokenizer: a speakerless caller's cloud egress must
    still go through local sanitization rather than skip it entirely.

    HA first (has tools for weather, time, devices), cloud as fallback for
    reasoning, the local base model (adapter-off, no history, no facts, no
    speaker context) as the final fallback when a local model is loaded and
    both hops fail or are unavailable — mirroring the documented
    HA → cloud → local-base-model fallback chain
    (:mod:`paramem.server.inference` module docstring).  In server-wide
    cloud-only mode there is no local model, so the final fallback is the
    canned limited-mode response instead.

    The cloud leg goes through :func:`~paramem.server.inference.
    answer_via_cloud` — the sole cloud-egress funnel.  With ``model``/
    ``tokenizer`` both ``None`` (cloud-only mode) it selects the
    cannot-anonymize branch: no local model means no ParaMem-held knowledge
    to protect on this path, so the current turn egresses verbatim and
    history is still drop-gated.  With a live ``model``/``tokenizer``
    (local mode, ``identity_absent=True``) it selects the normal
    ``sanitization.cloud_mode`` policy — the SAME anonymize/block/both
    policy every other leg applies — keyed off ``is_personal`` computed
    below from :func:`~paramem.server.sanitizer.is_self_referential`.  No
    intent classification (routing requires a resolved ``speaker_id``, so
    there is no ``RoutingPlan`` to consult here), and — same as every
    other leg — no speaker name or id reaches the cloud system prompt.

    When ``identity_absent`` is True and the turn is itself a personal
    interrogative (:func:`~paramem.server.inference._is_personal_interrogative`
    — the SAME shared predicate the abstention gate uses, over the
    ``is_self_referential`` verdict computed here), this returns the
    canned no-identity response BEFORE the HA leg is even tried — there is
    no speaker for a personal question to be about, and asking HA does not
    change that.  This short-circuit is NEVER gated on
    ``config.abstention.enabled`` (see
    :func:`~paramem.server.inference._is_personal_interrogative`'s
    docstring): refusing here is a structural impossibility (no identity,
    no store), not a feature toggle.  Non-personal and declarative turns
    fall through to the normal HA → cloud → base-model dispatch below,
    unaffected by ``identity_absent`` beyond the ``is_personal`` verdict
    threaded into the cloud leg.

    Args:
        text: The user's turn, verbatim.
        history: Prior conversation turns, or ``None``.  Drop-gated by
            :func:`~paramem.server.inference._sanitize_history` (inside
            ``answer_via_cloud``) before it egresses.  Callers pass ``[]``
            for a speakerless turn — no history egress at all, per the
            ``ServingPath.RELAY`` contract.
        config: The live :class:`~paramem.server.config.ServerConfig`.
        cloud_permitted: Whether the CLOUD leg may be used.  Computed by the
            caller (``_run_chat_turn``) from ``_state["cloud_only_reason"]``
            and ``config.cloud.allow_degraded_serving``: ``False`` closes the
            cloud leg while the server is cloud-only for an involuntary
            reason and the operator has not opted into degraded serving.  The
            HA leg is unaffected.
        ha_client: HA client, or ``None`` when HA is not configured.
        cloud_agent: Cloud agent, or ``None`` when cloud is not configured.
        language: Resolved BCP-47 code for this turn, or ``None``.
        speaker_id: Resolved canonical speaker ID, or ``None`` for a
            speakerless turn.  Live production consumer:
            :func:`~paramem.graph.flows.anonymize_turn` (via
            ``answer_via_cloud``'s anonymize branch) when ``model``/
            ``tokenizer`` are live — always ``None`` on this path since
            ``identity_absent`` is what selects that branch.
        identity_absent: ``True`` when this turn carries no resolved speaker
            at all (``ServingPath.RELAY``) — gates the no-identity
            short-circuit described above.  ``False`` (default) preserves
            the original server-wide-cloud-only behavior unchanged.
        model: The live base model for the local-mode relay leg (anonymize
            branch + base-model fallback), or ``None`` in server-wide
            cloud-only mode (no local model exists).
        tokenizer: Paired with *model*; ``None`` under the same condition.

    Returns:
        The answering leg's :class:`~paramem.server.inference.ChatResult`, or
        the canned limited-mode result when no leg served the turn.
    """
    is_personal_turn = False
    if identity_absent:
        from paramem.server.inference import _is_personal_interrogative

        is_personal_turn = is_self_referential(
            text, personal_referent_config=config.personal_referent
        )
        if _is_personal_interrogative(text, config, is_personal=is_personal_turn):
            logger.info("Relay route: no-identity short-circuit (personal interrogative)")
            return ChatResult(text=config.abstention.load_no_identity_response())

    # Try HA conversation agent — it has tools and real-time data
    # Language passed via HA's native conversation API parameter
    if ha_client is not None:
        logger.debug("Relay route: trying HA agent for: %s", text[:100])
        ha_languages = config.tools.ha.supported_languages
        response_text = ha_client.conversation_process(
            text,
            agent_id=config.ha_agent_id,
            language=language,
            supported_languages=ha_languages,
        )
        if response_text is not None:
            logger.info("Relay route: HA agent responded")
            return ChatResult(text=response_text, escalated=True)
        logger.info("Relay route: HA agent failed, trying cloud")

    # HA failed or unavailable → try cloud for reasoning
    if cloud_agent is not None and not cloud_permitted:
        logger.warning(
            "Relay route: cloud leg closed (degraded serving; "
            "set cloud.allow_degraded_serving: true to open it)"
        )
    elif cloud_agent is not None:
        logger.info("Relay route: escalating to cloud")
        result = answer_via_cloud(
            text,
            cloud_agent,
            config,
            is_personal=is_personal_turn,
            model=model,
            tokenizer=tokenizer,
            speaker_id=speaker_id,
            history=history,
            language=language,
            cloud_permitted=cloud_permitted,
        )
        if result is not None and result.text:
            logger.info("Relay route: cloud responded")
            return result

    if model is not None and tokenizer is not None:
        # Final link of the documented fallback chain: the local base
        # model, adapter-off (inside ``_base_model_answer``), no history,
        # no recalled facts, no speaker context (``history=None``,
        # ``speaker_id=None``).  ``cloud_agent``/``ha_client`` are passed
        # as ``None`` so a base-model [ESCALATE] cannot re-open a hop that
        # already failed or was closed above (and cannot bypass the
        # ``cloud_permitted`` gate, which ``_maybe_escalate``'s internal
        # cloud call does not itself receive).
        logger.info("Relay route: HA and cloud unavailable — falling back to local base model")
        from paramem.server.inference import _base_model_answer

        return _base_model_answer(
            text,
            None,
            model,
            tokenizer,
            config,
            cloud_agent=None,
            ha_client=None,
            speaker=None,
            speaker_id=None,
            language=language,
            is_personal=False,
        )

    logger.warning("Relay route: all services failed")
    return ChatResult(
        # Caller-centric: describes what happened to THIS request (nothing
        # answered it), not the server's internal operating mode — the
        # caller has no use for the "limited mode" label.
        text="I can't answer that right now. Please try again shortly.",
        escalated=True,
    )


# --- Internal ---


def _is_full_cycle_due(config) -> bool:
    """Oldest-interim-age deadline gate — the mechanism that fires the full fold.

    Replaces the window-stamp identity gate with a content-driven signal that
    is robust to skipped ticks, restarts, and the ≥24h stamp-collapse bug:

    **The deadline:** the fold is due when the **oldest un-folded interim
    tier's age** ≥ the full consolidation period (``N × refresh_cadence``
    seconds, ``N = max_interim_count``).  Anchored to the oldest interim, not
    to wall-clock since the last fold, so a skipped tick or a restart does not
    shift it.

    There is no second, count-based signal.  A slot-count gate ("due on the
    (N+1)-th slot") is unreachable by construction: ``consolidation_period_
    seconds`` is *derived* as ``refresh_cadence × N`` (``server/config.py``),
    so the deadline always lands inside the very window in which the (N+1)-th
    slot would be minted — that tick resolves FULL and drains the ring, so an
    (N+1)-th slot never comes into existence for a count gate to observe.

    **What is counted:** only interim slots that carry a written payload, in
    either venue (``iter_interim_dirs(..., payload_only=True)`` — a slot
    candidate is present, whether it carries ``graph.json`` or adapter
    weights; the gate stops asking which venue it is in).  A slot directory
    whose payload write never landed holds nothing to fold, so it must not
    drive the gate on its own.  ``_oldest_interim_stamp``,
    ``_full_cycle_deadline_dt`` and ``_seconds_until_next_full_consolidation``
    filter the same way — the gate, its deadline, its incident dedup key and
    ``/status``'s ETA must all describe one set.

    Gated on ``n ≥ 1`` (un-folded interims must exist).  An empty store
    returns ``False`` and stays on the interim path.

    **N == 0 special case (full-fold-only consume-pending mode)**: when
    ``max_interim_count == 0`` there are no interim adapter slots — the
    interim path never mints any, so ``n`` is always 0 and the count/deadline
    logic below does not apply.  Every scheduled tick IS a full-cycle tick at
    ``count == 0``; there is no interim path to route to.  This function does
    not decide whether there is anything to train on — that "no pending
    sessions → don't dispatch" decision is the arbitrator's content gate
    (:func:`~paramem.server.consolidation_action.consolidation_content_gate`,
    evaluated on the resolved action after this function returns), which
    returns ``noop_no_pending`` /
    ``noop_no_named`` before the fold is ever entered.  At ``N == 0`` this
    function itself returns ``True`` unconditionally.

    ``N < 0`` is rejected at config load; the defensive belt below guards
    against any future code path that bypasses the validator.

    Timestamps are in LOCAL time throughout, consistent with
    ``current_interim_stamp``'s ``datetime.now()`` basis.

    ``window_stamp`` is not read here, nor anywhere else.  ``write_tier_slot``
    still writes it on every main slot's manifest (derived via
    ``current_full_consolidation_stamp`` and threaded through
    ``run_build_and_publish`` / ``stage_event``'s ledger stamp), but purely
    as provenance — no code compares stamps to decide whether a fold is due.
    """
    from paramem.memory.interim_adapter import iter_interim_dirs

    N = config.consolidation.max_interim_count
    # count==0 → full-fold-only consume-pending mode: every tick is a full cycle.
    # There are no interim dirs to count or age; return True unconditionally so
    # the dispatcher routes every tick to the full path.  The noop machinery
    # in _run_full_cycle owns the "nothing to train" fast-path.
    if N == 0:
        return True
    # Negative N is config-rejected upstream; defensive belt only.
    if N < 0:
        return False

    if not any(iter_interim_dirs(config.adapter_dir, payload_only=True)):
        return False

    # Deadline: the oldest un-folded interim may not age past the full period.
    deadline_dt = _full_cycle_deadline_dt(config)
    if deadline_dt is None:
        return False
    return datetime.now() >= deadline_dt


def _oldest_interim_stamp(config) -> "str | None":
    """Return the oldest un-folded interim stamp, or ``None`` when none exist.

    Reads ``iter_interim_dirs(config.adapter_dir, payload_only=True)``,
    sorted ascending (oldest first), and extracts the timestamp from the directory
    name via ``interim_stamp_from_name``.  No age gate — returns the stamp regardless
    of how old it is.  Payload-less slot dirs are skipped: this stamp is the cycle key
    for the gate's incidents, so it must describe the same set ``_is_full_cycle_due``
    counts.

    Used as the per-cycle dedup key for ``record_incident`` calls that fire
    when the ring is full (``interim_cap_reached``, ``interim_overflow_pending``)
    or when the full fold has missed its runway (``full_consolidation_overdue``).
    The stamp is stable for the entire time that cycle's interims remain
    un-folded; it changes when a new cycle's interims become the oldest, so one
    incident fires per stuck cycle and reopens on a new cycle's failure.

    Timestamps are in LOCAL time throughout, consistent with
    ``current_interim_stamp``'s ``datetime.now()`` basis.
    A malformed internal stamp raises ``ValueError`` (no defensive try/except —
    a corrupt internal stamp should surface, not be silenced).

    Returns ``None`` when no interim directories exist (store is empty or the
    ring was just drained by the full fold).
    """
    from paramem.memory.interim_adapter import interim_stamp_from_name, iter_interim_dirs

    dirs = list(iter_interim_dirs(config.adapter_dir, payload_only=True))
    if not dirs:
        return None
    oldest_name, _ = dirs[0]  # sorted ascending; [0] is oldest
    stamp = interim_stamp_from_name(oldest_name)
    if stamp is None:
        raise ValueError(f"Corrupt interim slot name on disk: {oldest_name!r}")
    return stamp


def _full_cycle_deadline_dt(config) -> "datetime | None":
    """Return the full-fold deadline as a ``datetime``, or ``None``.

    The deadline is ``oldest_interim_dt + consolidation_period_seconds``.
    Returns ``None`` when:

    - No un-folded, payload-bearing interim slots exist (the scan is
      ``payload_only=True``, venue-blind, same set as :func:`_is_full_cycle_due`).
    - ``consolidation_period_seconds`` is ``None`` (manual-only cadence).

    Timestamps are in LOCAL time throughout, consistent with
    ``_is_full_cycle_due`` and ``_oldest_interim_stamp``.

    Single source of truth for both :func:`_is_full_cycle_due` (gate) and
    :func:`_seconds_until_next_full_consolidation` (predictor).
    """
    from paramem.memory.interim_adapter import (
        INTERIM_STAMP_FORMAT,
        interim_stamp_from_name,
        iter_interim_dirs,
    )

    full_period_seconds = config.consolidation.consolidation_period_seconds
    if full_period_seconds is None:
        return None
    dirs = list(iter_interim_dirs(config.adapter_dir, payload_only=True))
    if not dirs:
        return None
    oldest_name, _ = dirs[0]
    oldest_stamp = interim_stamp_from_name(oldest_name)
    if oldest_stamp is None:
        raise ValueError(f"Corrupt interim slot name on disk: {oldest_name!r}")
    oldest_dt = datetime.strptime(oldest_stamp, INTERIM_STAMP_FORMAT)
    return oldest_dt + timedelta(seconds=full_period_seconds)


def _seconds_until_next_full_consolidation(
    config,
    now: datetime | None = None,
) -> int | None:
    """Seconds until the next scheduled tick on which :func:`_is_full_cycle_due`
    would evaluate to ``True``.

    Based on the same gate logic as :func:`_is_full_cycle_due`:

    - ``N == 0`` (full-fold-only mode): every tick is a full fold — returns
      seconds to the next cadence boundary (same as ``next_interim_seconds``).
    - Cadence disabled (``refresh_cadence`` maps to ``None``): returns ``None``
      (manual-only, no scheduled ticks).
    - No payload-bearing interim slots exist yet: returns ``None`` (nothing to
      fold).  The scan is ``payload_only=True`` (venue-blind) exactly as the
      gate's is — an unfiltered predictor would contradict the gate in
      ``/status``.
    - Otherwise: ``deadline = oldest_interim_dt + full_period_seconds``; the
      result is the deadline ceiled to the next tick boundary, since folds only
      happen on ticks.

    ``consolidation_period_seconds is None`` (cadence disabled but N > 0) also
    returns ``None``.

    Parameters
    ----------
    config:
        Live server config.
    now:
        Override for "current time" (for testing).  Defaults to
        ``datetime.now()``.
    """
    from paramem.memory.interim_adapter import iter_interim_dirs
    from paramem.server.schedule_grammar import compute_schedule_period_seconds

    if now is None:
        now = datetime.now()

    N = config.consolidation.max_interim_count
    _refresh_seconds = compute_schedule_period_seconds(config.consolidation.refresh_cadence)
    if not _refresh_seconds or _refresh_seconds <= 0:
        return None  # manual-only — no scheduled ticks

    def _next_tick_seconds() -> int:
        """Seconds until the next cadence boundary from midnight."""
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        since_mid = int((now - midnight).total_seconds())
        next_boundary = ((since_mid // _refresh_seconds) + 1) * _refresh_seconds
        return max(0, next_boundary - since_mid)

    # N == 0: every tick is a full fold.
    if N == 0:
        return _next_tick_seconds()

    if not any(iter_interim_dirs(config.adapter_dir, payload_only=True)):
        return None  # no interims yet — nothing to fold

    deadline_dt = _full_cycle_deadline_dt(config)
    if deadline_dt is None:
        return None  # cadence off or no interims → no deadline

    if deadline_dt <= now:
        # Deadline already passed → due at next tick.
        return _next_tick_seconds()

    # Ceil the deadline to the next tick boundary.  Ticks fire at multiples
    # of _refresh_seconds from midnight of the deadline's day.
    deadline_midnight = deadline_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    deadline_since_mid = int((deadline_dt - deadline_midnight).total_seconds())
    k = deadline_since_mid // _refresh_seconds
    if deadline_since_mid % _refresh_seconds == 0:
        # Deadline lands exactly on a tick boundary — use that tick.
        tick_from_mid = k * _refresh_seconds
    else:
        # Ceil to the next tick.
        tick_from_mid = (k + 1) * _refresh_seconds
    tick_dt = deadline_midnight + timedelta(seconds=tick_from_mid)
    return max(0, int((tick_dt - now).total_seconds()))


def _overflow_incident_for(cycle_mode: str, overflow_slot: bool) -> "tuple[str, str] | None":
    """Return ``(incident_type, severity)`` for ring-cap states, or ``None``.

    Pure mapping with no I/O — single source of truth for the two emission
    sites in ``_extract_and_start_training`` that fire loud incidents when
    the interim ring reaches capacity.

    Args:
        cycle_mode: The ``"mode"`` value from ``run_consolidation_cycle``'s
            return dict (e.g. ``"cap_pending"``, ``"trained"``, ``"noop"``).
        overflow_slot: Whether the cycle minted an overflow slot
            (``result.get("overflow_slot", False)``).  Meaningful only when
            ``cycle_mode == "trained"``.

    Returns:
        ``("interim_overflow_pending", "failed")`` when the ring AND overflow
        are both exhausted (``cap_pending``).
        ``("interim_cap_reached", "warning")`` when an overflow slot was
        minted (ring full, but slack permitted one more adapter).
        ``None`` for all other outcomes (normal mint, noop, aborted).
    """
    if cycle_mode == "cap_pending":
        return ("interim_overflow_pending", "failed")
    if overflow_slot:
        return ("interim_cap_reached", "warning")
    return None


def _full_consolidation_overdue_key(config) -> "str | None":
    """Return the oldest un-folded interim stamp when the full fold missed its runway.

    The full fold is dispatched when due (oldest interim ≈ 1× full_period via the
    deadline backstop) and has one full period of runway before its next cycle
    boundary.  If the oldest interim reaches 2× full_period and has still not been
    drained, the fold missed its runway — that is severely wrong.

    Returns the oldest interim stamp (used as the per-cycle dedup key for
    ``record_incident``) when the overdue condition is met, otherwise ``None``.

    Cycle-key semantics: the oldest un-folded interim stamp is stable for the
    entire time that cycle's interims are un-folded.  It changes only when a new
    cycle's interims become the oldest — so one incident fires per stuck cycle and
    reopens on a new cycle's failure.

    Timestamps are in LOCAL time throughout, consistent with
    ``current_interim_stamp``'s ``datetime.now()`` basis.
    A malformed internal stamp raises ``ValueError`` (no defensive try/except —
    a corrupt internal stamp should surface, not be silenced).
    """
    from datetime import datetime

    N = config.consolidation.max_interim_count
    if N <= 0:
        return None
    period = config.consolidation.consolidation_period_seconds
    if period is None:
        # manual-only: no deadline, no overdue concept.
        return None
    oldest_stamp = _oldest_interim_stamp(config)
    if oldest_stamp is None:
        return None
    age = (datetime.now() - datetime.strptime(oldest_stamp, "%Y%m%dT%H%M")).total_seconds()
    return oldest_stamp if age >= 2 * period else None


def _record_full_consolidation_overdue(config) -> None:
    """Fire the ``full_consolidation_overdue`` incident when due.

    The one overdue check + incident, shared by the resume-pending-first arm
    (:func:`_dispatch_resume`) and the FULL dispatch arm of
    :func:`_dispatch_consolidation` — both need the identical key, summary
    and detail so the same stuck cycle is reported once regardless of which
    arm happens to run it.

    Purely informational: the fold still dispatches either way.  Deduped by
    the oldest un-folded interim stamp (:func:`_full_consolidation_overdue_key`)
    — stable for the entire time that cycle's interims are un-folded, so one
    incident fires per stuck cycle and reopens only on a new cycle's failure.

    A RECONCILE dispatch is exempt from firing this check: the schedule does
    not wait on a manually triggered reconcile the way it waits on a full
    fold, so the incident's own deduping (one open incident per stuck cycle)
    stays keyed to the schedule's own runway.  A reconcile still absorbs and
    reaps the interim ring like any full fold once dispatched — see
    ``_finalize_full``'s own incident resolution. The caller is responsible
    for not calling this when the action being dispatched is RECONCILE.
    """
    overdue_key = _full_consolidation_overdue_key(config)
    if overdue_key is None:
        return
    logger.warning(
        "Consolidation dispatch: full consolidation OVERDUE — oldest interim %s "
        "has aged >= 2x the full period without being folded",
        overdue_key,
    )
    record_incident(
        data_state_dir(config.paths.data),
        type="full_consolidation_overdue",
        key=overdue_key,
        severity="failed",
        summary="Full consolidation overdue — fold has not completed within its runway",
        detail={
            "oldest_interim_stamp": overdue_key,
            "type": "full_consolidation_overdue",
        },
    )


def _store_quarantine_verdict() -> "str | None":
    """``"deferred_store_quarantined"`` while the memory store is quarantined, else ``None``.

    The quarantine arm of :func:`refusal_for`'s closed verdict vocabulary —
    composed at the specific call sites that must refuse while the boot/lift
    store step has no publishable store (:func:`_hydrate_memory_store_in_place`):
    the consolidation arbitrator (:func:`_dispatch_consolidation`), the
    ``/migration/confirm`` · ``/migration/accept`` trial-state-transition
    doors, ``POST /speaker/forget``, and ``POST /interim/discard`` — every
    door that reads or writes the live ``MemoryStore``.

    In the arbitrator specifically, this is no longer an unconditional
    refusal while quarantined: a PENDING event resumes ahead of this check
    (:func:`_dispatch_consolidation`'s docstring, step 3) and — a resume
    needing nothing from the live store — completes and, on going fully
    live, lifts the quarantine itself (the heal at
    :func:`_finish_resumed_event`'s tail). This function's own verdict is
    unchanged for every other case: quarantined with nothing pending still
    defers here exactly as before, and a still-quarantined dispatch (no
    pending record, or a resume whose lift failed) reaches this check and
    refuses same as always.

    Deliberately NOT folded into :func:`_consolidation_dispatch_guards` (the
    shared predicate :func:`active_consolidation` wraps): that predicate
    also gates two doors that must keep serving, or keep repairing, while
    the store is quarantined — ``POST /debug/erase-keys`` (a file surgeon
    that touches neither the live store nor a model, and on a quarantined
    store attempts the lift itself after its file mutation) and
    ``POST /backup/restore`` / the base-swap branch of
    ``POST /migration/rollback`` (both recovery doors: quarantine is
    deliberately entered before the tree rewrite and exited by the lift on
    success, so refusing on an existing quarantine would make the door
    unable to recover from the very condition it exists to fix) — adding an
    arm here would close all three. See :func:`_consolidation_dispatch_guards`'s
    own tier-unverified comment for the identical reasoning applied to a
    sibling exception.

    Returns:
        ``"deferred_store_quarantined"`` when ``_state["store_quarantine"]``
        is set, else ``None``.
    """
    return "deferred_store_quarantined" if _state.get("store_quarantine") is not None else None


def _consolidation_dispatch_guards(*, include_cloud_only: bool = True) -> "str | None":
    """Shared pre-dispatch guard for consolidation dispatch.

    Checks the cross-cutting block conditions that prevent any consolidation
    fold from starting, for every action:

    - A base-swap migration is actively running
      (``_state["migration"]["base_swap_active"]``).
    - ``_state["consolidating"]`` — another fold is already running.
    - ``_state["mode"] != "local"`` — cloud-only mode (no model loaded).
      Skipped when *include_cloud_only* is ``False`` — the composition two
      doors use: ``POST /debug/erase-keys`` (a file-only repair door that
      needs no resident model) and ``POST /backup/restore`` (a RECOVERY
      door that must stay open in cloud-only mode — a cloud-only server is
      exactly a state a restore may need to run in). Cloud-only must not
      block either while every other busy arm still does.
    - Background trainer is actively training (GPU lock contention risk).
    - A migration TRIAL is active (:func:`_trial_active`) — the same refusal
      ``require_no_trial`` enforces at the REST boundary (HTTP 409 on every
      consolidation route), mirrored here for in-process callers that never
      go through FastAPI's dependency resolution.

    The base-swap check is a belt, not the buckle: on every production path
    ``base_swap_active`` is set only after ``migration["state"]`` is already
    ``"TRIAL"`` (``POST /migration/confirm`` sets ``TRIAL`` before the
    orchestration sets the flag; boot recovery sets ``TRIAL`` before
    launching the resumed orchestration; the sole reset — orchestration
    completion — clears both together), so ``_trial_active()`` below already
    refuses in practice. This check is explicit in case the flag ever lags
    the state — the same "in practice the state=TRIAL check fires first"
    framing ``/migration/confirm``'s own belt guard uses.

    Args:
        include_cloud_only: When ``False``, the cloud-only arm is skipped —
            every other arm (base-swap, already-running, bg-training,
            trial-active) still applies. ``True`` (default) preserves the
            full five-arm check every other caller relies on.

    Returns:
        A non-None ``"deferred_*"`` reason string when a block is in effect,
        ``None`` when clear (caller should proceed).  The string mirrors the
        vocabulary used by :func:`_dispatch_consolidation`.
    """
    if (_state.get("migration") or {}).get("base_swap_active", False):
        return "deferred_base_swap_active"
    if _state["consolidating"]:
        return "deferred_already_running"
    if include_cloud_only and _state["mode"] != "local":
        return "deferred_cloud_only"
    bg = _state.get("background_trainer")
    if bg is not None and bg.is_training:
        return "deferred_bg_training"
    if _trial_active():
        return "deferred_trial_active"
    return None


def _pending_event_state_dir(config) -> Path:
    """The directory a pending stage ledger would live under.

    ``data_state_dir(config.paths.data)`` is the directory
    ``ConsolidationLoop._fold_state_dir`` resolves to for the production
    loop — the only reason the arbitrator and the doors can ask the
    ledger's head without constructing a loop.
    """
    return data_state_dir(config.paths.data)


def _pending_event_action_name(config) -> "str | None":
    """The pending event's reported action name, or ``None`` when none is pending.

    Reporting reads ``event`` directly off the ledger head — ``"interim"``,
    ``"full"``, or ``"reconcile"`` — rather than deriving it: a report names
    the door.
    """
    from paramem.training import stage_ledger as _sl

    state_dir = _pending_event_state_dir(config)
    ledger = _sl.read_ledger(state_dir)
    if ledger is None:
        return None
    return ledger.event


def active_consolidation(*, include_cloud_only: bool = True) -> "str | None":
    """The activity predicate: the five busy arms, then the pending-record arm.

    Every externally-triggerable operation that mutates tier state or
    dispatches a consolidation reads this one function.  Order is the
    contract: a running event past its staging phase holds both arms —
    ``_state["consolidating"]`` was set at dispatch, and its ledger appeared
    inside the executor job — and the two verdicts differ, so answering the
    five busy arms first is what makes ``POST /reconsolidate`` against a
    running fold answer ``deferred_already_running`` instead of resuming an
    event that is already running.

    Args:
        include_cloud_only: Forwarded to
            :func:`_consolidation_dispatch_guards` — ``False`` for a door
            that mutates on-disk tier state without touching the live store
            or model (``POST /debug/erase-keys``), so cloud-only mode alone
            does not close it while every other busy/pending arm still
            does. ``POST /backup/restore`` calls
            :func:`_consolidation_dispatch_guards` directly instead of this
            function — the RECOVERY door must stay open in cloud-only mode
            AND on a pending consolidation event record, so only the
            five-arm guard applies there.

    Returns:
        The verdict key (one of ``_consolidation_dispatch_guards``'s five,
        or ``"deferred_event_pending"``), or ``None`` when no consolidation
        is active.
    """
    guard = _consolidation_dispatch_guards(include_cloud_only=include_cloud_only)
    if guard is not None:
        return guard
    config = _state.get("config")
    if _pending_event_action_name(config) is not None:
        return "deferred_event_pending"
    return None


def refusal_for(verdict: str, *, doing: str, then: str) -> "tuple[str, str]":
    """The verdict map: one ``(error, message)`` pair per verdict of
    :func:`active_consolidation`.

    Args:
        verdict: The string :func:`active_consolidation` (or
            :func:`_consolidation_dispatch_guards`) returned.
        doing: The gerund the three wait-shaped busy arms (in-flight,
            bg-training, base-swap) and the pending arm share —
            ``"forgetting a speaker"``, ``"discarding the interim ring"``,
            ``"erasing keys"``, ``"re-attributing orphan sessions"``,
            ``"cancelling queued ingest sessions"`` — rendered as
            ``"... before <doing>."``.
        then: The imperative tail the cloud-only arm takes — ``"forget"``,
            ``"discard"``, ``"erase"``, ``"re-attribute"``, ``"cancel"`` —
            rendered as ``"Reacquire the GPU (POST /gpu/acquire), then
            <then>."``.

    Returns:
        ``(error_code, message)``.  The five busy verdicts keep the codes
        their doors already answered with — a programmatic surface, not
        prose (``cli/backup_restore.py`` branches on ``detail["error"]``).
        The pending-record verdict is a distinct code,
        ``"consolidation_pending"``, so an operator or a script can tell
        "wait" from "finish now".  ``"deferred_store_quarantined"``
        (:func:`_store_quarantine_verdict`) maps to ``"store_quarantined"``
        and names the quarantine cause in the message.
    """
    if verdict == "deferred_store_quarantined":
        cause = (_state.get("store_quarantine") or {}).get("cause") or {}
        return (
            "store_quarantined",
            "The memory store is quarantined "
            f"({cause.get('exception_type', 'unknown')}: "
            f"{cause.get('message', 'unknown cause')}); wait for a repair before {doing}.",
        )
    if verdict == "deferred_event_pending":
        action_name = _pending_event_action_name(_state.get("config")) or "a run"
        return (
            "consolidation_pending",
            f"A pending consolidation event ({action_name}) is being resumed; "
            f"wait before {doing}. It clears at the next scheduled consolidation, "
            "or POST /consolidate to finish it now. A run that keeps failing to "
            "resume is superseded by restoring a healthy backup "
            "(POST /backup/restore).",
        )
    if verdict == "deferred_trial_active":
        return (
            "trial_active",
            "A migration TRIAL is in progress. Accept or roll back the migration first.",
        )
    if verdict == "deferred_cloud_only":
        return (
            "cloud_only",
            "Server is in cloud-only mode; no local model or consolidation loop is "
            f"available. Reacquire the GPU (POST /gpu/acquire), then {then}.",
        )
    if verdict == "deferred_already_running":
        return ("consolidating", f"Consolidation is running; wait for completion before {doing}.")
    if verdict == "deferred_bg_training":
        return (
            "training_active",
            f"Background training is active; wait for completion before {doing}.",
        )
    if verdict == "deferred_base_swap_active":
        return (
            "base_swap_active",
            f"A base-swap migration is actively running. Wait for it to complete "
            f"(or fail) before {doing}.",
        )


def _stamp_scheduled_run(config) -> None:
    """Record this dispatch's schedule mark as the last scheduled-run attempt.

    Stamps every real cadence kind — anchored (daily/weekly/HH:MM),
    exact-divisor intervals, and non-exact intervals alike — via
    :func:`~paramem.server.schedule_grammar.scheduled_run_stamp_value`, which
    writes the cadence's own calendar mark (or, for a non-exact interval, its
    heartbeat-floored stamp) rather than raw ``time.time()``: a
    second dispatch inside the same mark's window must read
    :attr:`~paramem.server.schedule_grammar.ScheduleDueStatus.NOT_DUE`, and an
    un-floored stamp would silently inflate a non-exact interval's effective
    period every cycle (see that function's docstring).

    No-op when the cadence is off or unparseable — there is no cadence to
    stamp a dispatch against, and
    :func:`~paramem.server.schedule_grammar.scheduled_run_stamp_value` raises
    ``ValueError`` for those inputs; this guards ahead of that call rather
    than catching the exception.

    Fires only on a SCHEDULED dispatch (the ``AUTO`` tick), on both the full and
    the interim path.  A manual run does not reset the cadence window: the
    scheduled tick keeps its content gate, so if the manual run already consumed
    everything the next tick noops on its own — cheaply, and without the manual
    run having to predict that.
    """
    from paramem.server import schedule_state as _schedule_state
    from paramem.server.schedule_grammar import parse_schedule_atom, scheduled_run_stamp_value

    cadence = config.consolidation.refresh_cadence or ""
    atom = parse_schedule_atom(cadence)
    if atom is None or atom.kind == "off":
        return
    _schedule_state.write_last_scheduled_run(
        data_state_dir(config.paths.data),
        scheduled_run_stamp_value(cadence, time.time()),
    )


def _dispatch_to_executor(
    fn: Callable[[], None],
    status: str,
    *,
    action: ConsolidationAction,
    spec: "calibrate_module.CalibrationRunSpec | None" = None,
) -> str:
    """Submit one run — a consolidation fold or a calibration probe — to the
    default executor and return *status*.

    The single dispatch ritual for every run this envelope submits.
    ``consolidating`` is set HERE — on the event-loop thread, before the
    executor submission — which is what makes the guard in
    :func:`_consolidation_dispatch_guards` free of a check-then-act race:
    every route is ``async def`` on that same loop, so no other dispatch can
    interleave between the guard's read and this write.  Structuring it in
    one place is the point: a hand-copied dispatch that forgets the flag
    would let two runs go concurrently, and the second would die in
    ``_ensure_staging_slot`` (``paramem/training/trainer.py``) or corrupt a
    calibration run's own artifact scope.

    Args:
        fn: The zero-arg sync entry point (``_extract_and_start_training``,
            ``_run_active_store_migration_sync``), or a ``functools.partial``
            that has already bound the entry point's arguments —
            ``_run_full_consolidation_sync`` takes its resolved door name
            (``event`` — ``"full"`` or ``"reconcile"``) that way, and
            ``_run_calibration_sync`` its ``CalibrationRunSpec``, since the
            executor contract itself carries no arguments.
        status: The status string to return to the caller on submission.
        action: The action this run resolved to — bound into the executor
            future's done callback (:func:`_consolidation_run_done`) so a
            crash records the right incident type.
        spec: The calibration run's own validated payload, when *action* is
            ``CALIBRATE``/``CALIBRATE_PENDING`` — bound into the same
            callback so a crash incident is keyed by the route and the
            run's own stamp.  ``None`` for every staging action.

    Returns:
        *status*, unchanged — so call sites read as ``return
        _dispatch_to_executor(fn, "started_full", action=action)``.
    """
    _state["consolidating"] = True
    # Use the loop captured once at lifespan startup (line ~2392) rather than
    # re-acquiring it here.  Every dispatch path is an ``async def`` route on
    # that same loop, so this is the identical loop in production — one source
    # of truth.  It also keeps the dispatch off the global
    # ``asyncio.get_running_loop`` seam, so tests inject their loop via
    # ``_state["event_loop"]`` instead of patching stdlib (which deadlocks
    # starlette's TestClient portal under anyio >= 4.14).  A missing
    # ``_state["event_loop"]`` is a real startup bug, so there is deliberately
    # no fallback.
    event_loop = _state["event_loop"]
    future = event_loop.run_in_executor(None, fn)
    future.add_done_callback(functools.partial(_consolidation_run_done, action, spec))
    return status


def _record_consolidation_resume_blocked_incident(exc, event: str) -> None:
    """Record the ``consolidation_resume_blocked`` incident for a refuse-and-hold
    resume outcome.

    Keyed by the tier that classified FOREIGN (:data:`~paramem.training.
    consolidation.FOREIGN`) — the ledger and every shadow artifact are left
    untouched by the resume routine itself; this is the operator-visible
    signal that a later dispatch will meet the same refusal.  It resolves
    when a later resume finally succeeds (the two finalizers' own
    successful completion), or when a wholesale tier rewrite discards the
    stuck record outright (``POST /backup/restore``'s snapshot-bundle
    restore, or an active-store migration).
    """
    try:
        record_incident(
            data_state_dir(_state["config"].paths.data),
            type="consolidation_resume_blocked",
            key=exc.tier,
            severity="failed",
            summary=f"Consolidation resume blocked: tier {exc.tier!r} classified {exc.reason!r}",
            detail={"event": event, "tier": exc.tier, "reason": exc.reason},
        )
    except Exception:
        logger.exception("Failed to record consolidation_resume_blocked incident (non-fatal)")


def _finish_resumed_event(loop, staged_event, *, router) -> dict:
    """Take a pending event's ledger the rest of the way live.

    Calls :meth:`~paramem.training.consolidation.ConsolidationLoop.run_build_and_publish`
    directly — skipping ``stage_event`` entirely, per its own contract for a
    resumed dispatch (a caller-reconstructed :class:`StagedEvent`
    naming the same ``state_dir``/``event``).  *router* is threaded only for
    a full event, mirroring the fresh-dispatch shape: the interim event's
    own finalizer owns its one reload.

    Returns a result dict shaped like the fresh-dispatch entries'
    (``run_consolidation_cycle`` / ``consolidate``) own — ``tiers_rebuilt``,
    ``consumed_session_ids``, ``consumed_episodic_rels``,
    ``consumed_procedural_rels``, ``completed``, ``aborted``,
    ``tier_bindings``, ``mode``, ``adapter_name`` — so the SAME finalizers
    (:func:`_finalize_interim` / :func:`_finalize_full`) consume it
    unchanged, including the publish verdict (``tier_bindings``,
    threaded straight from ``run_build_and_publish``'s own summary) their
    unverified-tier incident sweep now reads instead of re-walking the
    adapter tree.  The interim ``mode`` is
    computed through :func:`~paramem.training.consolidation.interim_outcome_label`
    — the same computation the fresh interim cycle uses — so a resumed event
    that aborted mid-bundle reports ``"aborted"`` rather than being
    mislabelled ``"noop"``.

    **Quarantine heal, at the tail.** Both venues traverse this ONE function,
    off the event loop on both, so the lift lives here rather than being
    duplicated per-venue. When the resumed event went fully live
    (``build_summary["all_live"]``) AND the store was quarantined coming
    into this call (``_state["store_quarantine"] is not None`` — the
    condition under which *loop* was constructed against a throwaway empty
    store, see :func:`_run_pending_event_resume`), this function calls
    :func:`_lift_quarantined_store` and, on success, REBINDS ``loop.store``
    (and re-derives its key-mint counters via
    :meth:`~paramem.training.consolidation.ConsolidationLoop._derive_key_counters`)
    to the freshly published ``_state["memory_store"]`` — in place, on the
    SAME loop object already cached at ``_state["consolidation_loop"]`` and
    already captured by the finalizer closure
    (:func:`_finalize_interim`/:func:`_finalize_full`) the caller is about
    to dispatch. This differs from every other lift caller (``POST
    /debug/erase-keys``, ``POST /backup/restore``), which instead null
    ``_state["consolidation_loop"]`` and let the next fold's lazily-cached
    loop recreate against the fresh store: nulling it here would leave the
    about-to-run finalizer holding a discarded, empty-store loop instead.

    **Venue-conditional lock around the lift.** The lift's source medium
    (:func:`_lift_quarantined_store` → :func:`_preload_memory_store` →
    :func:`_build_store_contents`) is selected by
    ``config.consolidation.mode`` — NOT by ``staged_event.venue`` — so a
    ``mode: train`` config reached via a ``disk``-venue ledger (a
    ``simulate``-staged event resumed after the operator switched the config
    to ``train``) drives the lift into a ``WeightMemorySource`` fill that
    calls ``model.generate()``. The weights venue's caller
    (:func:`_run_pending_event_resume` → :func:`_run_stage_b_cycle`) already
    holds the non-reentrant ``_gpu_thread_lock`` for the whole cycle
    (acquired by ``BackgroundTrainer``'s worker,
    ``paramem/server/gpu_lock.py:17`` via ``background_trainer.py:323``), so
    the weights-venue lift call stays bare — re-acquiring here would
    deadlock it. The disk venue holds no lock at all on entry, so its lift
    call takes ``gpu_lock_sync()`` itself (blocking, no timeout — mirrors the
    ``nullcontext() if lock_held else gpu_lock_sync()`` pattern elsewhere in
    this module) before calling :func:`_lift_quarantined_store`: without it,
    a mode-switched disk-venue resume could run ``model.generate()`` on the
    executor thread concurrently with a ``/chat`` turn holding the lock.

    Protected: a fault here must not skip the caller's finalizer/disposal —
    ledger disposal downstream is gated on ``result["completed"]``
    (``== build_summary["all_live"]``), which is already fixed by the time
    the lift runs, so a raising or ``False``-returning lift still lets the
    record dispose normally; a raising lift is caught and logged here
    (mirroring the "Protected: a fault here must not wedge the finalizer"
    idiom in :func:`_finalize_interim`), and a ``False``-returning lift
    needs no retry logic of its own — ``_lift_quarantined_store`` already
    recorded the fresh quarantine cause, and the NEXT dispatch simply defers
    on it again via :func:`_store_quarantine_verdict` (no pending record is
    left to resume).

    Raises:
        ConsolidationResumeBlocked: A not-yet-done tier classified FOREIGN.
            The ledger and every shadow artifact are left untouched.
    """
    from paramem.training import stage_ledger as _sl

    extraction_stage = _sl.extraction_entry(staged_event.ledger) or {}
    consumed_session_ids = list(extraction_stage.get("sessions", []))

    # No absorbed_interim_tiers to compute here: run_build_and_publish reads
    # it straight from the ledger (ledger.absorbed_interim_tiers, recorded
    # when the event was staged, phase 1) -- never recomputed from the live
    # store, which could disagree with what the original staging pass
    # actually absorbed if the ring already partially reaped before this
    # resume.
    build_summary = loop.run_build_and_publish(
        staged_event,
        router=router if _sl.full_topology(staged_event.event) else None,
    )

    result: dict = {
        "tiers_rebuilt": build_summary["published_tiers"],
        "consumed_session_ids": consumed_session_ids,
        "consumed_episodic_rels": extraction_stage.get("episodic_rels", 0),
        "consumed_procedural_rels": extraction_stage.get("procedural_rels", 0),
        "completed": build_summary["all_live"],
        "aborted": build_summary["aborted"],
        "tier_bindings": build_summary["tier_bindings"],
    }
    if not _sl.full_topology(staged_event.event):
        result["adapter_name"] = next(iter(staged_event.ledger.tiers), None)
        result["mode"] = interim_outcome_label(build_summary, venue=staged_event.venue)
        result["new_keys"] = []
        result["triples_extracted"] = 0

    # Quarantine heal: see this function's own docstring "Venue-conditional
    # lock around the lift" section above.  The weights venue already holds
    # _gpu_thread_lock for this whole cycle (re-acquiring would deadlock);
    # the disk venue holds none on entry, so it takes gpu_lock_sync() around
    # the lift itself -- the lift's source medium follows
    # config.consolidation.mode, not the venue, so a disk-venue ledger can
    # still resolve to a GPU-touching WeightMemorySource fill.  The lock
    # acquisition is inside the try so a raise there is caught the same as a
    # raise from the lift call.  Protected: a fault here must not skip the
    # caller's finalizer/disposal.
    if build_summary["all_live"] and _state.get("store_quarantine") is not None:
        config = _state["config"]
        try:
            if staged_event.venue == "disk":
                from paramem.server.gpu_lock import gpu_lock_sync

                with gpu_lock_sync():
                    lifted = _lift_quarantined_store(config)
            else:
                lifted = _lift_quarantined_store(config)
        except Exception:
            logger.exception(
                "Post-resume quarantine lift raised; store stays quarantined "
                "-- the resumed event still completes and disposes normally"
            )
            lifted = False
        if lifted:
            loop.store = _state["memory_store"]
            loop._derive_key_counters()
            # loop.promoted_keys is deliberately left as-constructed here
            # (seeded from store.iter_bookkeeping() at __init__ time, see
            # ConsolidationLoop.seed_key_metadata) -- not re-derived against
            # the rebound store. Traced benign: a promoted key has already
            # left the episodic active set by the time it was promoted, so
            # the rebound store's bookkeeping cannot un-promote it under
            # this loop instance, and re-flagging an already-promoted key as
            # semantic on a later cycle is idempotent.

    return result


def _run_pending_event_resume() -> None:
    """Executor entry point: resume a pending consolidation event straight
    from its stage ledger.

    The resume-pending-first arm of :func:`_dispatch_consolidation` submits
    this exactly like any fresh dispatch (:func:`_dispatch_to_executor` has
    already set ``_state["consolidating"] = True``).  The disk venue (a
    pending simulate event) needs no GPU lock and runs inline; the weights
    venue routes through :func:`_run_stage_b_cycle` for the same GPU-lock /
    cooldown-gate / crash-envelope every fresh training dispatch gets.

    A ``ConsolidationResumeBlocked`` reaching either path means the ledger
    and every shadow artifact are left untouched — recorded as its own
    ``consolidation_resume_blocked`` incident (never the generic crash
    incident a raise into ``_run_stage_b_cycle`` would otherwise produce),
    and the record stays pending for the next dispatch to meet the same
    refusal, until either a resume finally succeeds or a content-replacing
    operation whose write already overtook the record discards it (``POST
    /backup/restore``'s snapshot-bundle restore, or a base-swap rollback).
    An active-store migration is content-preserving, not content-replacing
    — it refuses outright, record untouched, while any record is pending.

    **Store-independent resume.** A pending event now resumes ahead of the
    store-quarantine verdict (:func:`_dispatch_consolidation`'s docstring,
    step 3), so this function may run with ``_state["memory_store"] is
    None``.  In that case it constructs a throwaway, locally-scoped empty
    :class:`~paramem.memory.store.MemoryStore` and threads it into whichever
    venue's loop-construction seam applies
    (:func:`~paramem.server.consolidation.get_or_create_consolidation_loop`'s
    *store* kwarg for the disk venue, :func:`_run_stage_b_cycle`'s *store*
    kwarg — same seam — for the weights venue) — ONLY when a fresh loop is
    being built; an
    already-cached process-lifetime loop ignores the override.  The resumed
    event needs nothing FROM the store: :meth:`~paramem.training.consolidation.
    ConsolidationLoop._derive_key_counters` scans an empty store down to the
    same donor floor a hydrated one would clamp at (harmless — the resumed
    event replays keys already decided at staging, not fresh mints), and
    :func:`~paramem.training.go_live.publish_bundle`'s
    ``store.adopt_increments`` reads only the bundle's own increment data,
    never prior store state.  :func:`_finish_resumed_event`'s own tail lifts
    the real store back online and rebinds the loop to it once this call
    completes successfully.  Every other caller of both seams passes no
    override and is unaffected.
    """
    from paramem.memory.store import MemoryStore
    from paramem.training import stage_ledger as _sl
    from paramem.training.consolidation import ConsolidationResumeBlocked, StagedEvent

    config = _state["config"]
    state_dir = data_state_dir(config.paths.data)
    ledger = _sl.read_ledger(state_dir)

    staged_event = StagedEvent(
        event=ledger.event,
        venue=ledger.venue,
        state_dir=state_dir,
        built_tiers=tuple(ledger.tiers),
        ledger=ledger,
    )

    # See the docstring's "Store-independent resume" section: None only
    # when the store is quarantined (or never yet built) coming into this
    # resume, which is exactly when this seam exists to unblock the heal.
    _store_override = MemoryStore() if _state.get("memory_store") is None else None

    if ledger.venue == "disk":
        loop = get_or_create_consolidation_loop(_state, store=_store_override)
        try:
            result = _finish_resumed_event(loop, staged_event, router=_state.get("router"))
        except ConsolidationResumeBlocked as exc:
            _record_consolidation_resume_blocked_incident(exc, ledger.event)
            _consolidation_terminal(None)
            return
        finalizer = (
            functools.partial(_finalize_interim, loop, result)
            if not _sl.full_topology(ledger.event)
            else functools.partial(_finalize_full, loop, result)
        )
        _consolidation_terminal(finalizer)
        return

    def _body(loop, bt) -> "tuple[str, Callable[[], None] | None]":
        try:
            result = _finish_resumed_event(loop, staged_event, router=_state.get("router"))
        except ConsolidationResumeBlocked as exc:
            _record_consolidation_resume_blocked_incident(exc, ledger.event)
            return "resume_blocked", None
        if not _sl.full_topology(ledger.event):
            return "resumed_interim", functools.partial(_finalize_interim, loop, result)
        return "resumed_full", functools.partial(_finalize_full, loop, result)

    _run_stage_b_cycle(
        kind="training_crash" if not _sl.full_topology(ledger.event) else "consolidation_crash",
        incident_key=ledger.event,
        failure_summary=f"Resumed {ledger.event} consolidation crashed unexpectedly",
        failure_detail={},
        body=_body,
        store=_store_override,
    )


def _dispatch_resume(config) -> "tuple[str, ConsolidationAction] | None":
    """Resume-pending-first: dispatch a pending event's resume, or ``None``
    when nothing is pending.

    Any consolidation dispatch that finds a pending ledger resumes and
    finishes that event before any new event starts.  The reported action
    derives from the ledger head's ``event`` field directly
    (:func:`_pending_event_action_name`) — ``"interim"`` is ``INTERIM``,
    ``"full"`` is ``FULL``, ``"reconcile"`` is ``RECONCILE``.  The
    content gate is skipped deliberately: a resume's input is the ledger,
    not new material.  The originally requested action waits for the next
    tick.
    """
    action_name = _pending_event_action_name(config)
    if action_name is None:
        return None
    resumed_action = {
        "interim": ConsolidationAction.INTERIM,
        "full": ConsolidationAction.FULL,
        "reconcile": ConsolidationAction.RECONCILE,
    }[action_name]
    if resumed_action is ConsolidationAction.FULL:
        _record_full_consolidation_overdue(config)
    logger.info(
        "Consolidation dispatch: resuming pending %s event ahead of any new dispatch",
        action_name,
    )
    return (
        _dispatch_to_executor(_run_pending_event_resume, "started_resume", action=resumed_action),
        resumed_action,
    )


def _dispatch_consolidation(
    action: ConsolidationAction,
    *,
    spec: "calibrate_module.CalibrationRunSpec | None" = None,
) -> "tuple[str, ConsolidationAction]":
    """Gate + dispatch one run against the model.  The single arbitrator.

    Every run that touches the model — a consolidation fold, or a
    calibration probe — comes through here: the systemd tick, every
    operator consolidation endpoint, and every ``/calibrate/*`` route.
    Nothing below this function knows who asked, beyond what *action* and
    *spec* say.

    ``AUTO`` is requested by ``/scheduled-tick`` and by the boot-completion
    catch-up task (:func:`_run_boot_completion_tasks`, which dispatches
    in-process rather than through a REST call) — no other caller resolves
    it, so ``action is ConsolidationAction.AUTO`` IS "this is a scheduled or
    boot-catch-up tick", not a separate flag threaded alongside it.
    ``/consolidate`` requests ``FULL`` directly, ``/consolidate/interim``
    requests ``INTERIM`` directly, ``/reconsolidate`` requests ``RECONCILE``
    directly — none of them ever resolves ``AUTO``, so none of them consults
    the deadline math or moves the cadence window; a manual door drops only
    the TIME condition (is a cycle due), never the CONTENT condition (is
    there anything to consume), which the content gate still enforces on the
    resolved-or-direct ``FULL``/``INTERIM``/``RECONCILE`` either way — each on
    its own input.  A ``/calibrate/*`` route requests ``CALIBRATE`` or
    ``CALIBRATE_PENDING`` and passes its own validated *spec* — the only
    caller that ever passes one; every other door passes ``None``.

    Order (unconditional gates first, so an explicit request cannot walk past a
    safety property; ★ = gated on ``action.stages_event``):

    1. ``_consolidation_dispatch_guards()`` — base-swap active / already-running
       / cloud-only / bg-training / migration TRIAL active.  All actions.
       Verified side-effect-free, so it can run before anything a pending
       event below might need.
    2. **Idle debounce** — all actions.  This protects a live chat turn from a
       long GPU seizure; it is a safety property, not a schedule, so an
       explicit request defers on it too.  MUST stay ahead of resume (step 3):
       a weights-venue resume trains on GPU, and a chat-turn abort landing
       mid-resume would livelock the very heal step 3 exists to run.
    3. ★ **Resume-pending-first** (:func:`_dispatch_resume`) — a pending
       event's ledger is resumed and finished before any new STAGING event
       starts, the identical contract for every staging action including
       ``RECONCILE``.  A non-staging action (a calibration probe) skips this
       step and proceeds straight to step 4: a calibration run writes no
       ledger, no shadow tree and no slot, and retires nothing — the resume
       replays shadow bytes recorded at staging time, which a probe's own
       dispatch neither depends on nor would corrupt.  A staging dispatch
       that resumes never reaches steps 4-7 on this same call — skipping the
       pre-stages (step 6-7) on a resuming dispatch is a deliberate, accepted
       behavior change: retiring orphan sessions is not time-critical, and
       the next non-resuming dispatch runs it.
    4. :func:`_store_quarantine_verdict` — the memory store is quarantined
       (the boot/lift store step could not publish a fresh
       :class:`~paramem.memory.store.MemoryStore`).  All actions, including
       ``RECONCILE`` and every calibrate action: with no store there is
       nothing to fold into, rebuild from, or construct the process-lifetime
       loop against.
    5. ★ **Any MAIN tier's registry binding unverified SINCE the last store
       step** — every STAGING action, including ``RECONCILE``.  A
       non-staging run mints no key and rewrites no registry, so the gate
       does not apply to it.  Distinct from step 4: this is drift a fold's
       own post-cycle revalidation (:func:`_revalidate_adapter_manifests`)
       observed AFTER the last successful boot/lift store step, not (yet)
       caught by a fresh one.
    6. **Retroactive orphan-session voice claim** (:func:`_retro_claim_orphan_sessions`)
       — every action, including both calibrate actions: it attributes, it
       never retires, so a probe benefits from the same attribution a fold
       does.
    7. ★ **Retiring triage** — :func:`classify_pending_sessions` (pure) runs
       unconditionally, alongside step 6, so the counts it returns are
       available to the content gate for every action; the retirement side
       effect (:func:`retire_unattributable_sessions`) runs only for a
       staging action.  Reached only on a dispatch that did not resume at
       step 3.
    8. ``pending_rehydration`` — an incoherent active store pre-empts every
       action, calibrate included, until the migration completes
       (``started_migration``).  Runs after resume (step 3) for the same
       reason it always did: the store migration is content-preserving and
       needs a coherent, record-free tree, so a pending event always resumes
       to completion first and the migration only ever reaches a dispatch
       with no pending record.
    9. **``AUTO`` only** — the suspend/power-off catch-up gate, and the
       resolution to ``FULL`` or ``INTERIM`` via :func:`_is_full_cycle_due`
       (its only call site).  Both belong to the schedule; every other
       action skips straight past them.
    10. ``noop_no_interim_tier`` — ``INTERIM`` only.
    11. **Every action reaching this point** —
        :func:`~paramem.server.consolidation_action.consolidation_content_gate`.
        An empty input set is empty whether the schedule resolved into it,
        an operator named it directly, or a calibration probe asked for the
        pending set.  A ``noop_*`` status is not a refusal — it is the
        answer.  ``CALIBRATE`` never noops (the operator already supplied
        the artifact); ``CALIBRATE_PENDING`` noops exactly as ``INTERIM``
        does.
    12. ★ :func:`_stamp_scheduled_run` — ``AUTO`` only (``AUTO`` is a staging
        action, so both conditions hold; the stamp is written where it is
        today).
    13. Dispatch table: ``FULL``/``RECONCILE`` → :func:`_run_full_consolidation_sync`;
        ``INTERIM`` → :func:`_extract_and_start_training`; ``CALIBRATE``/
        ``CALIBRATE_PENDING`` → :func:`_run_calibration_sync`, answering
        ``"started_calibration"`` — the one status that means THIS run was
        submitted.
    14. ★ The overdue incident :func:`_record_full_consolidation_overdue`
        stays inside the ``FULL`` arm.

    Args:
        action: ``AUTO`` (the scheduled tick — let ``_is_full_cycle_due``
            decide, ``/scheduled-tick`` only), ``FULL`` (collapse the interim
            slots into main now — resolved from ``AUTO`` or requested
            directly by ``/consolidate``), ``INTERIM`` (absorb pending
            sessions into a new interim slot — resolved from ``AUTO`` or
            requested directly by ``/consolidate/interim``), ``RECONCILE``
            (a full consolidation whose input excludes pending sessions:
            pending sessions stay pending; stored interim knowledge is
            absorbed and reaped like any full fold — ``/reconsolidate``),
            ``CALIBRATE`` (an operator-supplied calibration artifact), or
            ``CALIBRATE_PENDING`` (a calibration probe over the pending
            NAMED session set — ``POST /calibrate/extract_pending``).
        spec: The calibration run's own validated payload.  Present only on
            a ``CALIBRATE``/``CALIBRATE_PENDING`` dispatch; every other
            caller passes ``None``.

    Returns:
        ``(status, action)`` — the terminal status string and the action as
        resolved (the requested action when the dispatch never got as far as
        resolving ``AUTO``).  ``deferred_*`` means "blocked, retry next tick";
        ``noop_*`` means "nothing to do"; ``started*`` means the run was
        submitted to the executor.
    """
    config = _state["config"]
    _scheduled = action is ConsolidationAction.AUTO

    _guard = _consolidation_dispatch_guards()
    if _guard is not None:
        if _guard == "deferred_cloud_only":
            logger.info(
                "Consolidation dispatch (%s): mode=%s (reason=%s) — deferred, "
                "will retry on next tick",
                action.value,
                _state["mode"],
                _state.get("cloud_only_reason"),
            )
        elif _guard == "deferred_bg_training":
            logger.info(
                "Consolidation dispatch (%s): background training active — deferred", action.value
            )
        elif _guard == "deferred_trial_active":
            logger.info(
                "Consolidation dispatch (%s): migration TRIAL active — deferred", action.value
            )
        elif _guard == "deferred_base_swap_active":
            logger.info(
                "Consolidation dispatch (%s): base-swap migration active — deferred", action.value
            )
        else:
            logger.info(
                "Consolidation dispatch (%s): consolidation already running — deferred",
                action.value,
            )
        return _guard, action

    # Idle debounce — every action.  A fold seizes the GPU for minutes; firing
    # one seconds after a chat turn would strand the next one.  Ahead of
    # resume (below) on purpose: a weights-venue resume trains on GPU too, and
    # this is the one property that must hold regardless of what is pending.
    debounce_s = config.consolidation.training_idle_debounce_s
    last_chat = _state.get("last_chat_monotonic")
    # Check last_chat first so MagicMock configs (tests that patch _state with
    # a minimal mock config) short-circuit before the int comparison fires.
    if last_chat is not None and debounce_s > 0 and (time.monotonic() - last_chat) < debounce_s:
        elapsed = time.monotonic() - last_chat
        logger.info(
            "Consolidation dispatch (%s): chat %.1fs ago < debounce %ds — deferred",
            action.value,
            elapsed,
            debounce_s,
        )
        return "deferred_idle", action

    # Resume-pending-first: a pending event's ledger is resumed and finished
    # before any new event starts.  Runs ahead of the store-quarantine
    # verdict and the tier-unverified gate below (see this function's own
    # docstring, step 3): a resume replays shadow-byte increments recorded
    # at staging time rather than re-staging, so neither gate's rationale
    # applies to it, and completing it is what heals a store quarantined by
    # a crashed publish on a cold-born tier — see the lift at
    # :func:`_finish_resumed_event`'s tail.  Also ahead of the migration
    # pre-empt further below: a dispatch that finds both a pending record
    # and an armed mode switch resumes the event, and the migration only
    # ever reaches a dispatch with no pending record.  The content gate is
    # skipped deliberately — a resume's input is the ledger, not new
    # material — and so are the pre-stages below (retiring orphan sessions
    # is not time-critical; the next non-resuming dispatch runs them).
    if action.stages_event:
        _resume = _dispatch_resume(config)
        if _resume is not None:
            if _scheduled:
                # The resumed run counts as this window's run: stamping here
                # means a later record-free tick inside the same window reads
                # not-due instead of starting a fresh fold. It does not throttle
                # resume itself, which runs unconditionally while a record is
                # pending, regardless of the stamp.
                _stamp_scheduled_run(config)
            return _resume

    _quarantine_verdict = _store_quarantine_verdict()
    if _quarantine_verdict is not None:
        logger.warning(
            "Consolidation dispatch (%s): memory store is quarantined — deferred", action.value
        )
        return _quarantine_verdict, action

    # A MAIN tier (episodic/semantic/procedural) whose registry<->manifest
    # binding could not be verified has an unknowable key set: the merger's
    # identity space spans every main tier, so an invisible tier's keys get
    # re-minted as duplicates, and both persist branches rewrite all three
    # main registries -- overwriting the very file preserved for recovery.
    # This is NOT the same fault as the store-quarantine check above: this
    # arm reads adapter_manifest_status, refreshed by fold finalizers
    # (_revalidate_adapter_manifests) whenever a fold's OWN post-save
    # revalidation finds a tier newly broken -- drift discovered strictly
    # BETWEEN two store steps, which _store_quarantine_verdict cannot see
    # (quarantine is set only when the boot/lift store step itself runs and
    # fails; it is not re-run on every dispatch). This defers every action,
    # including RECONCILE (which rebuilds all three main registries from the
    # store). It lives here rather than in _consolidation_dispatch_guards
    # because POST /debug/erase-keys must stay open while a tier is
    # unverified this way -- adding an arm to that shared predicate would
    # close it too. That door calls active_consolidation() (which wraps
    # _consolidation_dispatch_guards with the pending-record arm), never
    # this dispatcher; the predicate itself has exactly two direct callers
    # -- active_consolidation() and this dispatcher -- and this arm binds to
    # the dispatcher alone. POST /speaker/forget and POST /interim/discard
    # separately compose _store_quarantine_verdict() at their own call
    # sites (see there) -- unlike this arm, that composition is specific to
    # those two doors, not shared via either predicate function. A pending
    # resume (step 3 above) never reaches this gate: it is not re-staging,
    # so an unknowable tier's identity space is not at risk from it.
    # BINDING_ROW_STATUSES (paramem/server/manifest_status.py) is the single-
    # sourced set of row statuses this arm defers on -- it includes
    # "keys_without_slot" and "payload_mismatch" alongside the pre-existing
    # three, so a main tier holding active keys with no slot, or whose bound
    # slot's payload no longer matches its manifest digest, defers every
    # action exactly like a no-matching-slot or key-count-mismatched tier
    # already did.
    if action.stages_event:
        from paramem.server.manifest_status import BINDING_ROW_STATUSES
        from paramem.utils.tiers import MAIN_TIERS

        _manifest_status = _state.get("adapter_manifest_status", {})
        if any(
            _manifest_status.get(_tier, {}).get("status") in BINDING_ROW_STATUSES
            for _tier in MAIN_TIERS
        ):
            logger.warning(
                "Consolidation dispatch (%s): a main tier's registry binding "
                "is unverified — deferred",
                action.value,
            )
            return "deferred_tier_unverified", action

    # Retroactive voice-match claim: scan orphan sessions against every
    # enrolled speaker. Attributes sessions whose embeddings match an
    # existing profile at high confidence. Cheap — centroids are cached.
    # Every action, including both calibrate actions: it attributes, it
    # never retires.  Not reached on a staging dispatch that resumed above
    # (see step 3's note).
    _retro_claim_orphan_sessions()

    # Pending-session triage: classify_pending_sessions is pure and runs
    # unconditionally, alongside the retro-claim above, so its counts feed
    # the content gate for every action including both calibrate actions.
    # The retirement side effect (retiring what can never be attributed, and
    # expired holdables) is a staging-only act: a non-staging run mutates no
    # session state.
    _triage = classify_pending_sessions(
        config,
        _state["session_buffer"],
        _state.get("speaker_store"),
    )
    if action.stages_event:
        retire_unattributable_sessions(config, _state["session_buffer"], _triage.drop_ids)
    _pending_count, _named_count = _triage.pending_count, _triage.named_count

    # Active-store migration gate: when a mode-switch was detected at startup
    # (or an in-progress migration was interrupted), every consolidation
    # dispatch routes to the migration sync until all tiers have cleared the
    # 1.0 recall gate. This pre-empts the action's own gates because the active
    # store is not yet coherent with the operator's yaml mode -- grouped with
    # the safety gates above, ahead of the schedule's own business below, for
    # the same reason resume-pending-first is. Runs AFTER resume-pending-
    # first (above): the store migration is content-preserving and needs a
    # coherent, record-free tree, so a pending event always resumes to
    # completion first and the migration runs on a later dispatch.
    if _state.get("pending_rehydration", False):
        if _state.get("integrity_check_failed", False):
            logger.warning(
                "Consolidation dispatch: active-store migration pending but "
                "integrity_check_failed=True — refusing to dispatch (boot-time integrity "
                "check failed; resolve the corrupt registry file and restart the server "
                "to retry)"
            )
            return "migration_skipped_degraded", action
        logger.info("Consolidation dispatch: active-store migration pending — running migration")
        return (
            _dispatch_to_executor(
                _run_active_store_migration_sync, "started_migration", action=action
            ),
            action,
        )

    # AUTO is requested only by /scheduled-tick and the boot-completion
    # catch-up task, so "action is AUTO" already identifies one of those two
    # -- no separate flag alongside it (see this function's docstring).  A
    # direct FULL/INTERIM/RECONCILE request skips this whole block: the
    # catch-up gate and the deadline resolution are the SCHEDULE's business,
    # never a manual door's.
    if _scheduled:
        # Suspend/power-off catch-up gate, universal across every real cadence
        # kind (anchored daily/weekly/HH:MM and exact-divisor intervals, not
        # just non-exact "heartbeat" intervals): the durable last-ATTEMPT
        # stamp (schedule_state.py) decides whether THIS tick actually
        # dispatches, via schedule_grammar's own dueness math
        # (scheduled_run_due). A cadence with no real schedule (off or
        # unparseable) has no mark to be due against and is never stamped —
        # it falls straight through to the deadline resolution below.
        from paramem.server import schedule_state as _schedule_state
        from paramem.server.schedule_grammar import ScheduleDueStatus as _ScheduleDueStatus
        from paramem.server.schedule_grammar import parse_schedule_atom as _parse_schedule_atom
        from paramem.server.schedule_grammar import scheduled_run_due as _scheduled_run_due

        cadence = config.consolidation.refresh_cadence or ""
        _cadence_atom = _parse_schedule_atom(cadence)
        if _cadence_atom is not None and _cadence_atom.kind != "off":
            last_attempt = _schedule_state.read_last_scheduled_run(
                data_state_dir(config.paths.data)
            )
            _due_status = _scheduled_run_due(cadence, last_attempt)
            if _due_status is _ScheduleDueStatus.NO_STAMP:
                _stamp_scheduled_run(config)
                logger.info(
                    "Scheduler tick: seeding catch-up stamp for cadence %r — "
                    "not dispatching this tick",
                    cadence,
                )
                return "noop_scheduler_seeded", action
            if _due_status is _ScheduleDueStatus.NOT_DUE:
                return "noop_not_due", action

        action = (
            ConsolidationAction.FULL if _is_full_cycle_due(config) else ConsolidationAction.INTERIM
        )

    # An interim tick mints an episodic_interim_* slot.  At max_interim_count==0
    # there is no interim tier to mint into — AUTO can never resolve here
    # (_is_full_cycle_due returns True unconditionally at N==0), and an explicit
    # request for a tier that does not exist is meaningless, not a fold to run.
    if action is ConsolidationAction.INTERIM and config.consolidation.max_interim_count == 0:
        logger.info(
            "Consolidation dispatch: interim requested but max_interim_count==0 "
            "(no interim tier exists) — noop"
        )
        return "noop_no_interim_tier", action

    # The content gate applies to every action reaching this point (FULL,
    # INTERIM, RECONCILE, CALIBRATE, or CALIBRATE_PENDING — resolved from
    # AUTO or requested directly): a manual door drops only the TIME
    # condition (the deadline math above), never the CONTENT condition here.
    # Each action reads its own input through the gate's own branch — see
    # `consolidation_content_gate`.
    _gate_status = consolidation_content_gate(
        action,
        config,
        pending_count=_pending_count,
        named_count=_named_count,
        memory_store=_state.get("memory_store"),
    )
    if _gate_status is not None:
        return _gate_status, action

    if _scheduled:
        # This tick is going to dispatch, so it consumes its cadence window.  A
        # manual run does not: the next scheduled tick keeps its own content
        # gate and noops by itself if the manual run already took everything.
        _stamp_scheduled_run(config)

    if action in (ConsolidationAction.FULL, ConsolidationAction.RECONCILE):
        logger.info("Consolidation dispatch: running the full fold (%s)", action.value)
        # Do NOT clear last_consolidation_error here.  A failure row stays visible
        # until the next op of that type SUCCEEDS (auto-resolved in the success paths
        # of the fold itself).  Clear-on-attempt hid still-failing conditions.

        # Overdue check: fire a loud incident when the prior cycle's full fold
        # missed its runway.  RECONCILE is exempt from firing it here, since
        # its own dispatch is not what the schedule waits on — but it absorbs
        # and reaps the interim ring exactly like a full fold once dispatched
        # (see _finalize_full's own incident resolution).
        if action is ConsolidationAction.FULL:
            _record_full_consolidation_overdue(config)

        # A manually triggered fold keys its telemetry and outputs exactly like
        # a scheduled one — same stamp, same family of rows.  There is no
        # manual flavour of a fold; the only thing the action decides below
        # the arbitrator is the door name recorded in the ledger — RECONCILE
        # runs the identical full-topology fold, differing only in leaving
        # sessions pending (see _run_full_consolidation_sync).
        _event = "reconcile" if action is ConsolidationAction.RECONCILE else "full"
        return (
            _dispatch_to_executor(
                functools.partial(_run_full_consolidation_sync, _event),
                "started_full",
                action=action,
            ),
            action,
        )

    if action in (ConsolidationAction.CALIBRATE, ConsolidationAction.CALIBRATE_PENDING):
        assert spec is not None, (
            f"{action.value} dispatch reached the executor arm with spec=None — every "
            f"caller of a calibrate action must pass its own validated spec"
        )
        logger.info("Consolidation dispatch: starting calibration run (%s)", action.value)
        return (
            _dispatch_to_executor(
                functools.partial(_run_calibration_sync, spec),
                "started_calibration",
                action=action,
                spec=spec,
            ),
            action,
        )

    logger.info("Consolidation dispatch: starting interim extract + train")
    return (
        _dispatch_to_executor(_extract_and_start_training, "started", action=action),
        action,
    )


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

    Atomic ordering: construct/ensure NEW pair loaded → update
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
                if loaded:
                    _state["vram_components"]["stt"] = stt_gpu.vram_delta_bytes
                else:
                    logger.warning(
                        "_set_voice_pipeline_profile('gpu'): STT load failed; voice path degraded"
                    )
            tts_gpu = _state.get("tts_gpu")
            if config.tts.enabled and tts_gpu is not None and not tts_gpu.is_loaded:
                try:
                    with vram_measure("tts") as _tts_vm:
                        tts_gpu.load_all()
                    if tts_gpu.is_loaded:
                        _state["vram_components"]["tts"] = _tts_vm["delta"]
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

            # Atomic flip: box and mirrors point at CPU pair BEFORE unloading GPU pair.
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
                _state.get("vram_components", {}).pop("stt", None)

            if old_tts_gpu is not None:
                try:
                    old_tts_gpu.unload_all()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "_set_voice_pipeline_profile('cpu'): tts_gpu.unload_all() failed; "
                        "continuing"
                    )
                _state["tts_gpu"] = None
                _state.get("vram_components", {}).pop("tts", None)

            safe_empty_cache()

    _state["voice_profile"] = profile
    logger.info("Voice pipeline profile: %r", profile)


def _end_voice_eviction(*, lock_held: bool) -> None:
    """Restore the voice pipeline to its target profile at a run's terminal.

    Idempotent, and therefore unconditional at every call site: on a run
    that never evicted this is a no-op, because ``_set_voice_pipeline_profile``
    is idempotent — the same property the executor done callback already
    relies on.

    *lock_held* is a property of the calling FRAME, not of the run:
    ``True`` on the ``BackgroundTrainer`` worker (which holds
    ``gpu_lock_sync()`` for the whole Stage-B phase, so a second acquisition
    would deadlock), ``False`` on the event loop and on an executor thread
    that has already released it.

    Terminal frames — three:

    - ``_run_stage_b_cycle``'s worker, wrapped around ``body(loop, bt)`` so
      it covers the success terminal and the crash terminal alike
      (``lock_held=True``).
    - The executor body of a run that ends without dispatching to Stage B —
      ``_extract_and_start_training``'s abort / no-facts / simulate arms,
      and ``_run_calibration_sync`` (``lock_held=False``).
    - ``_consolidation_run_done`` (the executor future's done callback)
      (``lock_held=False``).
    """
    try:
        _set_voice_pipeline_profile(_target_profile(), lock_held=lock_held)
    except Exception:
        logger.exception("Voice restore raised; ignoring")


def _build_bg_trainer(config) -> "BackgroundTrainer":
    """Construct a fresh BackgroundTrainer bound to the current model/tokenizer.

    The SINGLE source of the 6-arg BackgroundTrainer constructor literal.
    Reads ``_state["model"]`` and ``_state["tokenizer"]`` at call time so the
    returned instance is always bound to the live PeftModel wrapper.

    Does NOT store the result into ``_state`` — callers decide lifecycle.
    Use :func:`_active_bg_trainer` for the singleton get-or-create path;
    call this directly only at the migration rebuild sites that deliberately
    build a fresh trainer AFTER ``_release_base_model_in_process`` has already
    released the prior one — the Phase-A and Phase-B trainer rebuilds inside
    :func:`_run_base_swap_orchestration`, the only callers outside
    :func:`_active_bg_trainer`.

    Args:
        config: Live :class:`~paramem.server.config.ServerConfig` from
            ``_state`` (or ``config_b`` for Phase B migration).

    Returns:
        A new :class:`~paramem.server.background_trainer.BackgroundTrainer`
        instance bound to the current ``_state["model"]``/``["tokenizer"]``.
    """
    return BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
    )


def _active_bg_trainer(config) -> "BackgroundTrainer":
    """Return the process-lifetime singleton BackgroundTrainer, creating it on first call.

    Canonical accessor for the persistent trainer.  Every local-cycle dispatch
    site (boot replay, _await_bg_cycle, _extract_and_start_training,
    _run_full_consolidation_sync) must call this instead of constructing a new
    BackgroundTrainer — doing so avoids orphaning the prior worker thread, which
    would form a thread→bound-method→owner reference cycle that keeps the dead
    trainer's transient VRAM (~700 MiB/fold) alive across GC.

    On first call (``_state["background_trainer"] is None``): constructs via
    :func:`_build_bg_trainer` and stores the result on ``_state``.

    On subsequent calls: refreshes ``bt.model`` and ``bt.tokenizer`` from
    ``_state`` so the singleton always tracks the current PeftModel wrapper
    (create_interim_adapter may rebind ``_state["model"]`` between cycles).

    The migration rebuild sites — the Phase-A and Phase-B trainer rebuilds
    inside :func:`_run_base_swap_orchestration` — deliberately bypass this
    accessor and call :func:`_build_bg_trainer` directly: they build a FRESH
    trainer bound to a reloaded model AFTER ``_release_base_model_in_process``
    has already released the prior one via ``bt.release()``.  These are NOT leaks.

    Args:
        config: Live :class:`~paramem.server.config.ServerConfig` from
            ``_state``.

    Returns:
        The singleton :class:`~paramem.server.background_trainer.BackgroundTrainer`
        stored on ``_state["background_trainer"]``.
    """
    bt = _state.get("background_trainer")
    if bt is not None:
        # Refresh model/tokenizer handles in case create_interim_adapter
        # or a prior cycle rebound _state["model"] to a new PeftModel wrapper.
        bt.model = _state["model"]
        bt.tokenizer = _state["tokenizer"]
        return bt
    bt = _build_bg_trainer(config)
    _state["background_trainer"] = bt
    return bt


def _await_bg_cycle(
    *,
    loop,
    config,
    episodic_rels: list,
    procedural_rels: list,
    speaker_id: str,
    mode: "Literal['simulate', 'train']",
    run_label: str,
    pending: "PendingRelations",
    schedule: str = "",
    max_interim_count: int = 7,
    inference_fallback_adapter: str = "episodic",
    session_ids: "list[str]",
) -> dict:
    """Submit ``run_consolidation_cycle`` to the BG trainer and block until done.

    Fetches the process-lifetime singleton
    :class:`~paramem.server.background_trainer.BackgroundTrainer`
    via :func:`_active_bg_trainer`, submits the cycle as a callable, and blocks
    on a ``threading.Event`` until the worker finishes.  The BG worker thread
    acquires ``gpu_lock_sync()`` internally, so the caller must NOT already hold
    the lock (that would deadlock — the non-reentrant ``threading.Lock`` in
    ``gpu_lock.py`` cannot be acquired twice on the same thread).

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
        mode: ``"train"`` writes adapter weights; ``"simulate"`` writes a
            ``graph.json`` payload into the same written-slot envelope.
        run_label: Traceability tag passed to ``run_consolidation_cycle``.
        pending: The batch's merged extraction product — the caller's own
            :meth:`~paramem.training.consolidation.ConsolidationLoop.take_pending_relations`
            take, forwarded verbatim to ``run_consolidation_cycle``.
        schedule: Consolidation refresh-cadence string for stamp computation.
        max_interim_count: Cap on concurrent interim adapters.
        inference_fallback_adapter: Adapter name recorded for bookkeeping only
            on the sentinel job while the cycle runs — nothing reads it back
            to reactivate an adapter; inference generation always runs under
            ``model.disable_adapter()`` regardless.  Defaults to ``"episodic"``.
        session_ids: The app layer's own authoritative completed-session list
            (``extraction.completed_session_ids(session_buffer)``), forwarded
            verbatim to ``run_consolidation_cycle`` — the source of truth for
            what the resulting ledger's extraction stage retires.  Required:
            a relation-derived fallback would miss a session yielding zero
            relations or only attribute-typed facts.

    Returns:
        Result dict from ``run_consolidation_cycle``:
        ``{"triples_extracted", "new_keys", "adapter_name", "mode", "venue",
        "error"}``.

    Raises:
        Exception: Re-raises any exception thrown inside the BG worker.
    """
    bt = _active_bg_trainer(config)
    # Wire the singleton into the consolidation loop so /chat's abort_for_inference
    # targets the same BT instance whose training_hooks_for_job is installed
    # inside run_consolidation_cycle.  Without the loop wiring, the loop's
    # _build_training_hooks would close over a stale (or None) _bg_trainer and
    # silently drop the abort signal.
    loop._bg_trainer = bt

    _result_holder: dict = {}

    def _run() -> None:
        # T2b (simulate-fold): pre-task GPU cooldown gate — same category as
        # _run_interim_training and _run_full_cycle.  Simulate mode runs the
        # same GPU extraction chain; only the persist sink differs (graph.json
        # vs weights).  One call per fold kickoff on the BG worker thread.
        wait_for_cooldown(
            config.vram.cooldown_gate_threshold_c,
            config.vram.cooldown_gate_max_wait_fold_s,
            config.vram.cooldown_gate_poll_s,
            label="fold",
        )
        _result_holder["result"] = loop.run_consolidation_cycle(
            episodic_rels,
            procedural_rels,
            speaker_id=speaker_id,
            mode=mode,
            run_label=run_label,
            pending=pending,
            schedule=schedule,
            max_interim_count=max_interim_count,
            session_ids=session_ids,
        )

    bt.submit_and_wait(_run, inference_fallback_adapter=inference_fallback_adapter)
    return _result_holder["result"]


def _run_extraction_phase(
    loop,
    mark_sessions: bool = True,
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
    mark_sessions:
        When ``True`` (default), ``session_buffer.mark_consolidated`` is called
        after extraction/training (production behaviour).  Pass ``False`` for
        the trial path so pending sessions remain in the buffer — the
        transcript sweeper blocks archive+delete on pending sessions.

    Returns a result dict including the loop instance for reuse.

    Classification: all class-1 mock sites (prevent real execution only;
    ``_run_extraction_phase`` may close over ``_state`` directly).
    """
    import time

    from paramem.server.consolidation import (
        SessionClass,
        _dedup_episodic,
        _dedup_procedural,
        classify_session,
        session_retention_dir,
    )

    config = _state["config"]
    session_buffer = _state["session_buffer"]

    if "episodic" not in config.tier_config_map():
        logger.info("Episodic adapter is disabled in config, skipping consolidation")
        return {"status": "disabled", "sessions": 0, "loop": loop}

    start_time = time.time()

    # Filter pending to NAMED-only via classify_session.  The trial path
    # must not extract anonymous or embedding-only sessions — they have no
    # stable subject attribution and would corrupt the trial graph.
    speaker_store = _state.get("speaker_store")

    def _is_anon_ep(sid: str | None) -> bool:
        return bool(speaker_store is not None and sid and speaker_store.is_anonymous(sid))

    _named_ids_ep: set[str] = {
        f["session_id"]
        for f in session_buffer.pending_facts()
        if classify_session(
            speaker_id=f["speaker_id"],
            is_anonymous=_is_anon_ep(f["speaker_id"]),
            has_voice_embedding=f["has_voice_embedding"],
        )
        == SessionClass.NAMED
    }

    pending = [s for s in session_buffer.get_pending() if s["session_id"] in _named_ids_ep]
    if not pending:
        logger.info("No pending sessions to consolidate")
        return {"status": "no_pending", "sessions": 0, "loop": loop}

    logger.info("Consolidating %d NAMED pending sessions", len(pending))

    # --- Extract all sessions ---
    all_episodic_rels = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []
    enrichment_signals: list[dict] = []
    total_relations = 0

    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_speaker_id = session.get("speaker_id")

        session_ids.append(session_id)

        speaker_name = None
        if speaker_store is not None:
            speaker_name = speaker_store.resolve_speaker_name(session_speaker_id)
        with vram_scope(session_id):
            episodic_rels, procedural_rels = loop.extract_session(
                transcript,
                session_id,
                speaker_id=session_speaker_id,
                speaker_name=speaker_name,
                enrichment_provider=config.consolidation.extraction_enrichment_provider,
                enrichment_provider_model=config.consolidation.extraction_enrichment_provider_model,
                enrichment_provider_endpoint=(
                    config.consolidation.extraction_enrichment_provider_endpoint or None
                ),
                plausibility_judge=config.consolidation.extraction_plausibility_judge,
                plausibility_stage=config.consolidation.extraction_plausibility_stage,
                source_type=session.get("source_type", "transcript"),
                event_time=session["started_at"],
            )

        for rel in episodic_rels:
            rel["speaker_id"] = session_speaker_id
            # Stamp the real session id for uniform provenance plumbing.
            # Retention machinery (keep-pending) is NOT wired here — the trial
            # path runs mark_sessions=False so sessions cannot be lost regardless.
            rel["session_id"] = session_id
        for rel in procedural_rels:
            rel["speaker_id"] = session_speaker_id
            rel["session_id"] = session_id

        # This session's own enrichment signal — no incident is written
        # here; this function is the staging caller for the migration-trial
        # path (it runs its own extraction loop rather than routing through
        # the shared _extract_pending_sessions), so it arbitrates the batch
        # itself below, once extraction finishes — the same obligation
        # _extract_and_start_training and the full-cycle consume-pending
        # pre-stage each discharge right after their own extraction call
        # (see enrichment_signal's and arbitrate_enrichment_incidents's own
        # docstrings for why the read/write split this way).
        enrichment_signals.append(enrichment_signal(loop, session_id))

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

    # This function merges via extract_session directly (not through the
    # server's own _extract_pending_sessions), so this is where its own
    # extraction lifetime ends — the one take, ahead of either fold call
    # below.
    pending_relations = loop.take_pending_relations()

    # Staging-only bookkeeping the extraction stage itself no longer
    # performs: this batch's enrichment signals are this function's own to
    # arbitrate, exactly as _extract_and_start_training and the full
    # pre-stage each do right after their own extraction call.
    loop.arbitrate_enrichment_incidents(enrichment_signals)

    if not all_episodic_rels and not all_procedural_rels:
        logger.info("No relations extracted — skipping training")
        if mark_sessions:
            session_buffer.mark_consolidated(
                session_ids,
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
                pending=pending_relations,
                schedule=config.consolidation.refresh_cadence,
                max_interim_count=config.consolidation.max_interim_count,
                session_ids=session_ids,
            )
            # A "noop" result (no registry, or — not reachable here since
            # all_episodic_rels/all_procedural_rels are already known
            # non-empty by this point — no relations) means
            # run_consolidation_cycle returned before stage_event ever ran,
            # so no tier committed this cycle. Per-tier bookkeeping is written
            # only inside write_tier_slot / publish_tier_registry, at that
            # tier's own commit — there is no whole-store flush left to call
            # for a cycle that written no tier (the same "a cycle that writes
            # no tier does not advance its state" consequence documented on
            # cycle_count).  Every other outcome ("simulated") already
            # reached that bookkeeping write inside run_consolidation_cycle.
        except Exception:
            logger.exception(
                "Simulated consolidation failed — leaving %d sessions pending",
                len(session_ids),
            )
            raise

        if mark_sessions:
            session_buffer.mark_consolidated(
                session_ids,
                retention_dir=session_retention_dir(loop, config),
            )
        elapsed = time.time() - start_time
        summary = {
            "status": "simulated",
            "sessions": len(session_ids),
            "total_relations": total_relations,
            "episodic_rels": len(all_episodic_rels),
            "procedural_rels": len(all_procedural_rels),
            "episodic_keys": len(loop.store.active_keys_in_tier("episodic")),
            "semantic_keys": len(loop.store.active_keys_in_tier("semantic")),
            "procedural_keys": len(loop.store.active_keys_in_tier("procedural")),
            "elapsed_seconds": round(elapsed, 1),
            "simulated": sim_result.get("mode") == "simulated",
            "loop": loop,
        }
        logger.info("Simulation complete: %s", {k: v for k, v in summary.items() if k != "loop"})
        return summary

    # --- Train once ---
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
        from paramem.utils.vram_guard import vram_scope as _vram_scope

        with _vram_scope("training"):
            loop.run_consolidation_cycle(
                all_episodic_rels,
                all_procedural_rels,
                speaker_id=primary_speaker,
                mode="train",
                run_label=f"full-{primary_speaker or 'anon'}",
                pending=pending_relations,
                schedule=config.consolidation.refresh_cadence,
                max_interim_count=config.consolidation.max_interim_count,
                session_ids=session_ids,
            )
        # This branch is reached only from the base-swap trial path
        # (_run_extraction_phase's sole caller).  run_consolidation_cycle
        # above runs the interim fold with its own gate: training commits
        # into the cycle's own interim adapter slot
        # (episodic_interim_<stamp>) via write_tier_slot / publish_tier_registry,
        # gated by its staged-weights recall verdict (_probe_recall / _assert_tier_recall)
        # before that commit — see RecallGateRejected's docstring for the
        # two current raise sites.  The per-tier commit below is a separate
        # action on the trial's own loop: it copies the main
        # "episodic"/"semantic"/"procedural" PEFT adapters — resident,
        # unchanged by this event — into the trial tree via
        # ConsolidationLoop.commit_main_tiers, one commit_tier_slot
        # (paramem/memory/persistence.py) call per tier, stamping them with
        # the trial store's registries.  commit_tier_slot durably writes each
        # tier's own key_metadata.json rows as part of its own commit
        # sequence, so no separate bookkeeping write is needed after it.
        # This tier set is DIFFERENT from a production fold's
        # tiers_rebuilt by design: it is every resident main adapter
        # (unchanged by the trial event), not the subset a fold retrained.
        _trial_tiers = []
        if "episodic" in loop.model.peft_config:
            _trial_tiers.append("episodic")
        if "semantic" in loop.model.peft_config:
            _trial_tiers.append("semantic")
        if "procedural" in loop.model.peft_config:
            _trial_tiers.append("procedural")
        loop.commit_main_tiers(_trial_tiers, output_dir=loop.output_dir)
    except Exception:
        logger.exception(
            "Consolidation failed during train/save — leaving %d sessions pending",
            len(session_ids),
        )
        raise

    if mark_sessions:
        session_buffer.mark_consolidated(
            session_ids,
            retention_dir=session_retention_dir(loop, config),
        )

    elapsed = time.time() - start_time
    summary = {
        "status": "complete",
        "sessions": len(session_ids),
        "total_relations": total_relations,
        "episodic_keys": len(loop.store.active_keys_in_tier("episodic")),
        "semantic_keys": len(loop.store.active_keys_in_tier("semantic")),
        "procedural_keys": len(loop.store.active_keys_in_tier("procedural")),
        "elapsed_seconds": round(elapsed, 1),
        "loop": loop,
    }
    logger.info(
        "Consolidation complete: %s",
        {k: v for k, v in summary.items() if k != "loop"},
    )
    return summary


def _consolidation_run_done(
    action: ConsolidationAction,
    spec: "calibrate_module.CalibrationRunSpec | None",
    future,
) -> None:
    """Executor future's done callback for every run dispatched through
    :func:`_dispatch_to_executor`.  Clears the consolidating flag only if
    the run's OWN executor entry point raised — a normal return means the
    entry point has already dispatched (or completed) its own terminal.

    On success, the entry point (``_extract_and_start_training``,
    ``_run_full_consolidation_sync``, ``_run_active_store_migration_sync``,
    ``_run_calibration_sync``) has either submitted the next phase to the BG
    trainer (whose own terminal will clear the flag) or already called
    :func:`_consolidation_terminal` itself.

    On an uncaught exception this callback is the only terminal reached, so
    it records a typed incident, ends the voice eviction
    (:func:`_end_voice_eviction`, ``lock_held=False`` — the callback runs on
    the event-loop thread after the executor future has already returned;
    idempotent, safe on a run that never evicted), and calls
    :func:`_consolidation_terminal` itself:

    - ``VramExhausted`` — a durable ``vram_exhausted`` incident, so the
      failure is visible via ``/status`` without scraping logs.
    - Any other exception on a STAGING action — logged only (unchanged);
      the staging entry points raise into ``_run_stage_b_cycle``'s own
      crash envelope before ever reaching an uncaught exception here, so
      this arm is a defensive fallback, not the primary path.
    - Any other exception on a non-staging (calibration) action — a
      ``calibration_crash`` incident, keyed by the run's own route
      (``spec.route_path``), detailed with ``spec.run_id``.  This closes an
      existing gap: previously an uncaught calibration-executor exception
      produced no incident at all, only a log line.

    On a non-staging action (``spec is not None``) the terminal it dispatches
    ALSO sets ``_state["calibration_run"]["outcome"] = "crashed"`` (plus
    ``finished_at``), keyed by ``spec.run_id`` — independent of which
    incident branch fired above, so a VRAM-exhausted calibration crash
    records the outcome exactly like any other.  Without this, the record
    :func:`_submit_calibration_run` published at dispatch keeps
    ``outcome: None`` forever after a crash, since the normal-completion
    terminal (:func:`_run_calibration_sync`'s own) never runs; a client
    polling for the response then finds no ``response.json`` and no way to
    tell "still running" from "crashed" apart.  A staging action's terminal
    stays bookkeeping-free (``None``), as before — staging's own outcome
    lives in its ledger, not this record.

    Args:
        action: The action this run resolved to — bound at dispatch time.
        spec: The calibration run's own validated payload, or ``None`` for
            a staging action.
        future: The completed executor future.
    """
    exc = future.exception()
    if exc:
        logger.error("Consolidation run (%s) failed: %s", action.value, exc, exc_info=exc)
        if isinstance(exc, VramExhausted):
            phase = exc.args[0] if exc.args else "unknown"
            _at = datetime.now(timezone.utc).isoformat()
            record_incident(
                data_state_dir(_state["config"].paths.data),
                type="vram_exhausted",
                key=str(phase),
                severity="failed",
                summary=f"Consolidation: VRAM exhausted at phase {phase}",
                detail={"type": "vram_exhausted", "phase": phase, "at": _at},
            )
        elif not action.stages_event and spec is not None:
            record_incident(
                data_state_dir(_state["config"].paths.data),
                type="calibration_crash",
                key=spec.route_path,
                severity="failed",
                summary=f"Calibration run crashed: {spec.route_path}",
                detail={
                    "type": "calibration_crash",
                    "route_path": spec.route_path,
                    "run_id": spec.run_id,
                    "error": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )

        terminal_body = None
        if not action.stages_event and spec is not None:

            def terminal_body() -> None:
                _record = _state.get("calibration_run")
                if _record is not None and _record.get("run_id") == spec.run_id:
                    _record["outcome"] = "crashed"
                    _record["finished_at"] = datetime.now(timezone.utc).isoformat()

        _end_voice_eviction(lock_held=False)
        _consolidation_terminal(terminal_body)


def _run_stage_b_cycle(
    *,
    kind: str,
    incident_key: str,
    failure_summary: str,
    failure_detail: dict,
    body: "Callable[[object, object], tuple[str, Callable[[], None] | None]]",
    store=None,
) -> None:
    """Own the fire-and-forget BG-worker lifecycle shared by the interim-train,
    full-cycle, and active-store-migration Stage-B closures.

    Creates/reuses the process-lifetime :class:`ConsolidationLoop` (via
    :func:`~paramem.server.consolidation.get_or_create_consolidation_loop`), wires the singleton
    :class:`BackgroundTrainer`, then submits *body* to the trainer under the
    entry GPU-cooldown gate.  *body* receives ``(loop, bt)`` and must return
    a terminal descriptor ``(outcome, finalizer)`` or raise — it must never
    call ``_consolidation_terminal`` itself; this function is the sole dispatch
    point, reached from exactly one call site for every terminal (success or
    failure — there is no ``if mode`` fork here).

    On a normal return, the terminal's ``finalizer`` — a zero-arg callable,
    or ``None`` for a clear-only terminal — is dispatched via
    :func:`_consolidation_terminal`. ``_state["model"]`` needs no re-sync:
    the base model's object identity is fixed at load time, so *body*'s
    ``create_adapter`` / ``create_interim_adapter`` calls (via ``loop``)
    mutate the identical object already held there.

    ``body(loop, bt)`` is wrapped in ``_end_voice_eviction(lock_held=True)``
    so the voice pipeline is restored on the success terminal and the crash
    terminal alike, from this ONE call site — idempotent, safe on paths
    that never evicted voice.

    On an uncaught exception, records a *kind* incident keyed by
    *incident_key* (severity ``"failed"``) and dispatches
    :func:`_consolidation_terminal` with no bookkeeping.  A terminal that
    the caller wants to treat as a *normal*, non-incident outcome
    (``aborted``, ``noop``, ``extraction_failed``, ...) must be
    returned by *body*, not raised — raising always produces a *kind*
    incident.

    Args:
        kind: Incident type recorded on an uncaught exception
            (``"training_crash"``, ``"consolidation_crash"``, or
            ``"migration_error"``).
        incident_key: Incident dedup key — ``"interim"``, ``"active_store"``,
            or the full-topology door name recorded verbatim (``"full"`` or
            ``"reconcile"``).  Also used as the neutral cycle label on the
            success-path completion log — unlike ``kind``, it carries no
            incident-type implication.
        failure_summary: Human-readable incident summary on an uncaught
            exception.
        failure_detail: Incident detail payload on an uncaught exception.
        body: The path-specific Stage-B payload, closing over whatever
            pre-stage state (extracted relations, session ids, ...) it needs.
        store: Forwarded to :func:`~paramem.server.consolidation.get_or_create_consolidation_loop`
            unchanged — ``None`` (default) for the three ordinary Stage-B
            entry points; the pending-event resume's weights-venue call
            passes a locally-constructed empty store when
            ``_state["memory_store"] is None``, so a quarantined store
            never blocks the resume that heals it.
    """
    config = _state["config"]
    loop = get_or_create_consolidation_loop(_state, store=store)
    bt = _active_bg_trainer(config)
    loop._bg_trainer = bt

    def _worker() -> None:
        wait_for_cooldown(
            config.vram.cooldown_gate_threshold_c,
            config.vram.cooldown_gate_max_wait_fold_s,
            config.vram.cooldown_gate_poll_s,
            label="fold",
        )
        try:
            try:
                outcome, finalizer = body(loop, bt)
            finally:
                _end_voice_eviction(lock_held=True)
        except Exception as exc:
            logger.exception("%s crashed", kind)
            # BookkeepingInvariantViolation names the divergent tier/keys on
            # the exception itself (raised by the pre-write parity gate among
            # other boundaries); fold them into the incident detail so the
            # incident record — not just the log traceback — identifies what
            # diverged.  ActiveKeyHydrationFailure names the keys it could
            # not hydrate and the venue it tried.  RecallGateRejected reaches
            # here from either fold kind, main-tiers or interim — both route
            # a recall-gate rejection through this same crash path — and
            # names the tier that fell short of 100% recall over its own
            # full key set, and the individual keys that failed.  Every
            # other exception keeps the generic detail unchanged.
            from paramem.memory.store import BookkeepingInvariantViolation

            incident_detail = dict(failure_detail)
            if isinstance(exc, BookkeepingInvariantViolation):
                incident_detail["divergent_keys"] = exc.divergent_keys
            if isinstance(exc, ActiveKeyHydrationFailure):
                incident_detail["dropped_keys"] = exc.dropped_keys
                incident_detail["venue"] = exc.venue
            if isinstance(exc, RecallGateRejected):
                incident_detail["adapter_name"] = exc.adapter_name
                incident_detail["recall_rate"] = exc.recall_rate
                incident_detail["threshold"] = exc.threshold
                # Only publish failed_keys when non-empty.  Both current
                # RecallGateRejected raise sites populate it directly from
                # the failing probe entries (ConsolidationLoop._assert_tier_recall
                # in consolidation.py; _migrate_tier_simulate_to_train in
                # active_store_migration.py) — a rate that falls short of a
                # <=1.0 threshold always leaves at least one failed key, so
                # in practice this guard has nothing to filter today.  Kept
                # as a defensive no-op against a future raise site that
                # cannot supply per-key data.
                if exc.failed_keys:
                    incident_detail["failed_keys"] = exc.failed_keys
            try:
                record_incident(
                    data_state_dir(config.paths.data),
                    type=kind,
                    key=incident_key,
                    severity="failed",
                    summary=failure_summary,
                    detail=incident_detail,
                )
            except Exception:
                logger.exception("Failed to record %s incident (non-fatal)", kind)
            _consolidation_terminal(None)
            return
        # Neutral cycle label on the success line — `kind` is an incident
        # TYPE ("training_crash"/"consolidation_crash"/"migration_error")
        # and is reserved for the crash path; logging it here would misread
        # as an incident on a successful cycle.  `incident_key` ("interim"/
        # "full"/"active_store") identifies the Stage-B path without that
        # implication.
        logger.info("%s: cycle complete (outcome=%s)", incident_key, outcome)
        _consolidation_terminal(finalizer)

    bt.submit(_worker, inference_fallback_adapter="episodic")


# Interim-cycle outcomes that never represent a completed encoding attempt —
# ABORT (yielded to an inference request before training ran) and CAP_PENDING
# (interim ring full, session stays queued for the next full fold).  Neither
# is an encoding failure, so contributing sessions are pinned (kept pending)
# rather than retired — see the pin logic in ``_run_interim_training``. A
# recall-gate rejection is not in this set: it raises ``RecallGateRejected``
# and never reaches the pin logic at all — the crash path leaves the
# contributing sessions pending structurally, with nothing to pin.
_INTERIM_NON_ENCODING_OUTCOMES: frozenset[str] = frozenset({"aborted", "cap_pending"})


def _retire_ledger_sessions_and_dispose(loop, *, disposed: bool) -> list[str]:
    """Retire what the event's own ledger recorded, then dispose the record.

    Retirement reads the ledger fresh from disk — never a value captured
    earlier in the same process — so a crash between this call and the
    :func:`~paramem.training.stage_ledger.dispose` call it ends with leaves
    the SAME durable evidence on disk for the next resume to read:
    retire-then-dispose is what makes that resume idempotent (an
    already-retired session set retires again harmlessly, and
    disposal-of-an-absent-record is a no-op).  Disposing first — even when a
    copy of the session list survives in RAM — would delete that evidence: a
    crash in the gap strands the sessions pending with no pending record
    left for anything to resume.

    A retirement failure is convergence, not loss, and is treated the same
    way a crash is: the ledger is left pending (dispose is NOT called) so
    the next scheduled consolidation or boot resume re-reads the same
    ledger and retries retirement from it — idempotently, since a session
    already marked consolidated tolerates a repeat call.  The failure is
    recorded as a ``session_retirement_failed`` incident (session ids and
    cause) rather than only logged, so a stuck retirement stays visible
    until it resolves; a subsequent successful retirement resolves every
    active incident of that type, which also clears any instance stranded
    on a deployed store from before this event's own failure.

    *disposed* is the driver's own ``all_live`` verdict (every tier the
    ledger names verifies ``tier_live``).  ``False`` means this event is
    not actually complete (an abort or a partial bundle) — nothing is
    retired and the ledger stays pending for a genuine resume; the caller
    is expected to reach this only on outcomes that already imply
    completion, and the gate exists as the belt for that expectation
    rather than a routine branch.

    Returns:
        The session ids retired (possibly empty) — folded into the
        caller's own run-status detail.  Empty also on a retirement
        failure: nothing is retired-and-disposed on that path, so there is
        nothing this call can honestly report as retired.
    """
    if not disposed:
        return []

    from paramem.server.consolidation import session_retention_dir
    from paramem.training import stage_ledger as _sl

    ledger = _sl.read_ledger(loop._fold_state_dir)
    session_ids: list[str] = []
    if ledger is not None:
        extraction_stage = _sl.extraction_entry(ledger) or {}
        session_ids = list(extraction_stage.get("sessions", []))
        if session_ids:
            incidents_dir = data_state_dir(_state["config"].paths.data)
            try:
                _state["session_buffer"].mark_consolidated(
                    session_ids,
                    retention_dir=session_retention_dir(loop, _state["config"]),
                )
            except Exception as exc:
                logger.exception(
                    "Consolidation event: session retirement failed -- the ledger "
                    "stays pending; the next resume retries retirement from it"
                )
                record_incident(
                    incidents_dir,
                    type="session_retirement_failed",
                    key=f"{ledger.event}:{ledger.stamp}",
                    severity="failed",
                    summary="Session retirement failed after a completed consolidation event",
                    detail={"session_ids": session_ids, "cause": str(exc)},
                )
                return []
            resolve_incidents_by_type(incidents_dir, "session_retirement_failed")
    _sl.dispose(loop._fold_state_dir)
    return session_ids


def _finalize_interim(loop, result: dict, *, extraction=None) -> None:
    """Success/terminal finalizer for every interim-shaped consolidation
    event — the scheduled tick, in either venue, and a resumed pending
    event whose ledger names the interim door (``ledger.event`` is
    literally ``"interim"``).

    Runs on the asyncio event loop via ``_consolidation_terminal``.  Revalidates
    every tier's ``adapter_manifest_status`` row against the freshly-saved
    interim slot (:func:`_revalidate_adapter_manifests` — NOT pure: it also
    performs its documented ``.pending/`` sweep via ``sweep_orphan_pending``,
    see that function's own docstring; the read+hash cost profile otherwise
    matches the full-fold call this mirrors), records (or clears) any
    ``tier_registry_unverified`` incident for the tier(s) THIS event's
    ledger names (:func:`_record_unverified_tier_incidents` over
    ``result["tier_bindings"]`` — the publish verdict
    :meth:`~paramem.training.consolidation.ConsolidationLoop.run_build_and_publish`
    already computed, never a fresh tree walk; empty when the event did not
    reach ``all_live``, so an aborted/cap_pending terminal sweeps nothing),
    reloads the router, retires what the ledger recorded and disposes the
    event's record, records the durable run-status row, auto-resolves the
    incidents a clean interim success clears, and clears
    ``_state["consolidating"]``. Covers every non-crash interim terminal
    (``trained`` / ``simulated`` / ``cap_pending`` / ``aborted``) — a
    recall-gate rejection never reaches this finalizer at all, since
    ``RecallGateRejected`` propagates to the crash path instead.
    ``training_crash`` / ``vram_exhausted`` auto-resolution is gated on
    ``result["completed"]``: an ``aborted`` or ``cap_pending`` terminal made
    no encoding attempt, so it must not clear an operator's pending
    crash/exhaustion signal.

    The incident bookkeeping is wrapped in its own protected region: this
    is the SOLE operator-visible reporter for an unverified tier (the
    row-driven attention populator deliberately stays silent for it — see
    ``attention.py``), so a fault recording it must be logged, never allowed
    to wedge the finalizer before ``_state["consolidating"]`` clears.

    The no-staging terminal (``result["mode"] == "noop"`` with no
    ``adapter_name``) writes no ledger at all, so the ledger-based
    retirement above has no record to read: with *extraction* supplied,
    this pre-stage's own successfully-extracted sessions are retired here
    instead, through :func:`_retire_extracted_sessions` — the one call site
    for that fallback, shared by every caller that can reach this terminal
    (a fresh scheduled tick, either venue) — and its returned set is what
    gets reported, rather than the empty set the ledger-based retirement
    above always returns on this terminal.  A caller with no fresh
    extraction to fall back on (a resumed event) never reaches this
    terminal in practice — a resumed ledger always names an ``adapter_name``
    — so *extraction* stays ``None`` there.

    Args:
        loop: The cycle's ``ConsolidationLoop`` (post-training PEFT rebind).
        result: The ``run_consolidation_cycle`` return dict.  Relation
            counts for the run-status detail are read from
            ``result["consumed_episodic_rels"]`` /
            ``result["consumed_procedural_rels"]`` (the ledger's own
            extraction stage, recorded when the event was staged); the retired session list
            itself is re-read from the ledger at this finalizer's own
            retire-then-dispose step (see
            :func:`_retire_ledger_sessions_and_dispose`), never from a
            value captured earlier in the process, except on the
            no-staging terminal (see above).
        extraction: The pre-stage's own ``_PendingExtraction`` outcome, for
            the no-staging terminal's own retirement fallback.  ``None``
            (the default) when the caller has none to offer — safe exactly
            when the no-staging terminal cannot occur (a resumed event).
    """
    loop.model.eval()
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    # Re-validate manifests now that this interim slot has been freshly
    # saved with a new registry hash — mirrors _finalize_full's step c, so
    # a stale FINGERPRINT MISMATCH row for this interim doesn't linger on
    # /status until the next full cycle happens to prune it.
    _revalidate_adapter_manifests(_state)
    # The unverified-tier incident sweep consumes the publish verdict this
    # event's own driver call already computed (result["tier_bindings"] --
    # one verify_tier_binding read per tier the event's ledger names,
    # populated only when the event went all_live) rather than re-walking
    # the whole adapter tree.  This narrows what an interim fold observes
    # to the interim slot IT touched: a DIFFERENT tier's binding breaking
    # is no longer caught here -- it stays open until that tier's own next
    # publish (interim or full) or the next boot check, sound under the
    # single-writer architecture (tier state changes only at publish).
    # Protected: a fault here must not wedge the finalizer.
    _config = _state["config"]
    try:
        _record_unverified_tier_incidents(_config, result.get("tier_bindings", {}))
    except Exception:
        logger.exception(
            "Post-interim tier-incident bookkeeping failed (non-fatal); finalization continues"
        )
    _state["router"].reload()

    # Retire what the ledger recorded, then dispose the event's record.  A
    # crash between the two re-enters, finds every tier entry present,
    # retires an already-retired set (idempotent) and disposes.
    session_ids = _retire_ledger_sessions_and_dispose(loop, disposed=bool(result.get("completed")))

    # The no-staging terminal: no ledger was ever written (stage_event's own
    # no-material early exit), so the ledger-based retirement above always
    # returns [] here.  A "noop" that DOES carry an adapter_name reached a
    # real ledger that simply never went live (all_live False); that record
    # stays pending for the next resume and is intentionally left alone.
    _no_staging = result.get("mode") == "noop" and result.get("adapter_name") is None
    if _no_staging and extraction is not None:
        session_ids = _retire_extracted_sessions(
            extraction, _state["session_buffer"], loop, _config
        )

    # inference path sees the just-written interim slot (and any tier whose
    # format drifted from the loop's current setting).  Count via the
    # indexed_key_registry — it tracks every active key regardless of which
    # adapter (main or interim) currently holds it.  The previous
    # main-tier-simhash sum under-reported by the count of keys living in
    # episodic_interim_<stamp> slots between full cycles.
    total_keys = len(loop.store.all_active_keys())
    _interim_outcome = result.get("mode", "trained")
    _interim_detail = {
        "sessions": len(session_ids),
        "total_keys": total_keys,
        "adapter": result.get("adapter_name"),
        "episodic_rels": result.get("consumed_episodic_rels", 0),
        "procedural_rels": result.get("consumed_procedural_rels", 0),
    }
    try:
        record_last_run(
            data_state_dir(_state["config"].paths.data),
            op_type="consolidation",
            outcome=_interim_outcome,
            summary=(
                f"Interim {_interim_outcome}: adapter={result.get('adapter_name')}, "
                f"{total_keys} total keys"
            ),
            detail=_interim_detail,
        )
        # Auto-resolve op-level incidents cleared by a successful interim
        # cycle — gated on completion so an aborted/cap_pending terminal
        # (this finalizer is the terminal for every non-crash interim
        # outcome, not only "trained"/"simulated") leaves the operator's
        # crash/exhaustion signal in place.
        _interim_state_dir = data_state_dir(_state["config"].paths.data)
        if result.get("completed"):
            resolve_incidents_by_type(_interim_state_dir, "training_crash")
            resolve_incidents_by_type(_interim_state_dir, "vram_exhausted")
        # A refuse-and-hold resume that later completes (the FOREIGN write was
        # resolved out-of-band, or the record was replaced by a disposing
        # door and this is a fresh event) clears the incident it opened.
        resolve_incidents_by_type(_interim_state_dir, "consolidation_resume_blocked")
    except Exception:
        logger.exception("Post-interim run-status/incident bookkeeping failed (non-fatal)")
    logger.info(
        "Scheduled-tick complete — adapter=%s, %d total keys",
        result.get("adapter_name"),
        total_keys,
    )


@dataclass
class _PendingExtraction:
    """Outcome of one pending-session extraction stage.

    Returned by :func:`_extract_pending_sessions` to BOTH consolidation
    paths (interim tick and the full cycle's consume-pending pre-stage).
    The stage NEVER raises ``ExtractionFailed``: an abort is reported in
    *aborted* so the caller can restore the voice pipeline with the
    ``lock_held`` value its own lock context demands.  Raising would leave
    the caller no way to learn whether voice was evicted, and a
    ``finally``-based restore would fire with the GPU lock still held.

    Attributes
    ----------
    episodic_rels:
        Episodic relations from every session that extracted successfully,
        stamped with ``speaker_id`` / ``session_id`` provenance.
    procedural_rels:
        Procedural relations, same provenance stamping.
    session_ids:
        Every session the stage attempted, in extraction order (including
        the OOM-skipped ones — ``failed_session_ids`` is the filter).
    failed_session_ids:
        Sessions whose extraction exhausted VRAM.  They are NOT retired;
        they stay pending for the next cycle.  ALIASING CONTRACT: the interim
        caller binds this set by reference (``failed_session_ids =
        extraction.failed_session_ids``) and the pin logic mutates it in
        place after the stage returns — ``update(session_ids)`` for a
        non-encoding outcome (ABORT / CAP_PENDING) — with retirement reading
        the mutated set back through :meth:`completed_session_ids`.  Do NOT
        give this field a defensive copy; that would silently detach pinning
        from retirement.
    speaker_ids:
        Speaker id per successfully-extracted session (the interim path
        takes the last one as the cycle's primary speaker).
    evicted_voice:
        ``True`` when the batch contained a document session and the stage
        moved the voice pipeline to CPU.  The CALLER owns the restore.
    pending:
        The fold's input — this batch's merged extraction product, captured
        (and the extraction graph's keying surface reset) exactly once, at
        this stage's single return, by
        :meth:`~paramem.training.consolidation.ConsolidationLoop.take_pending_relations`.
    per_session:
        One record per session attempted — ``{session_id, speaker_id,
        source_type, episodic_count, procedural_count, status}`` — for the
        run's own diagnostics; never read by production retirement/fold
        logic.
    chunk_failures:
        This run's own OOM-skipped-chunk records — the same dicts a staging
        caller additionally copies into ``_state["chunk_failures"]``.
    enrichment_signals:
        One record per session — ``{session_id, anonymize,
        cloud_enrichment_degraded}`` — read from each session's own
        ``session_graph.diagnostics`` (via ``loop.last_session_graph``)
        right after its ``extract_session`` call.  No incident is written
        here; a staging caller passes this batch to
        :meth:`~paramem.training.consolidation.ConsolidationLoop.arbitrate_enrichment_incidents`.
    vram_headroom_warning:
        This run's own low-headroom attention record (the dict
        :func:`~paramem.utils.vram_guard.check_vram_headroom` writes into,
        never ``_state`` directly), or ``None`` when headroom never
        dropped below the configured floor.  A staging caller copies it
        into ``_state["vram_low_headroom_warning"]``.
    aborted:
        The ``ExtractionFailed`` that aborted the whole batch, or ``None``
        on a normal return.  When set, no session may be retired.
    """

    episodic_rels: list[dict]
    procedural_rels: list[dict]
    session_ids: list[str]
    failed_session_ids: set[str]
    speaker_ids: list[str]
    evicted_voice: bool
    pending: "PendingRelations"
    per_session: list[dict]
    chunk_failures: list[dict]
    enrichment_signals: list[dict]
    vram_headroom_warning: "dict | None"
    aborted: "ExtractionFailed | None" = None

    def completed_session_ids(self, session_buffer) -> list[str]:
        """Sessions that may be retired: extraction-succeeded AND retirable.

        Failed chunks stay pending so the next cycle retries them.  Document
        chunk sessions are held back unless ALL chunks of the same ``doc_id``
        completed in this cycle — partial success leaves the whole document
        pending (``SessionBuffer.retirable``).
        """
        raw = {sid for sid in self.session_ids if sid not in self.failed_session_ids}
        return session_buffer.retirable(raw)


def _retire_extracted_sessions(extraction, session_buffer, loop, config) -> "list[str]":
    """Retire the sessions this pre-stage successfully extracted, for an
    outcome that consumed them without producing an event record.

    An interim tick's own no-staging terminal
    (``stage_event``'s no-material early exit, surfaced as
    ``run_consolidation_cycle``'s ``"mode": "noop"`` / ``"completed": False``
    result with no ``adapter_name``) writes no ledger at all — the event
    terminal that normally retires via the ledger
    (:func:`_retire_ledger_sessions_and_dispose`) has nothing to read.  That
    early exit fires only when the batch extracted no relations AT ALL
    (``not episodic_rels and not procedural_rels``) AND no tier in this
    event's working universe already holds an active key — never merely
    because everything extracted was a duplicate of an existing fact: a
    batch of all-duplicates still has new material, so ``stage_event``
    still writes a ledger (and dedup runs, and finds, nothing new to write).
    The full path's own consume-pending pre-stage reaches the identical
    no-material shape.  Without this, the sessions this pre-stage extracted
    would re-extract on every tick forever, since no record exists for
    anything to retire them from.

    Args:
        extraction: The pre-stage's own ``_PendingExtraction`` outcome.
        session_buffer: The live ``SessionBuffer``.
        loop: The cycle's ``ConsolidationLoop`` — resolves the retention
            directory.
        config: The live ``ServerConfig``.

    Returns:
        The session ids retired — callers with no ledger to read this from
        (the shape this function exists for) report this set directly
        rather than under-reporting a retirement the ledger never recorded.
    """
    from paramem.server.consolidation import session_retention_dir

    retired = extraction.completed_session_ids(session_buffer)
    session_buffer.mark_consolidated(
        retired,
        retention_dir=session_retention_dir(loop, config),
    )
    return retired


def _extract_pending_sessions(loop, *, lock_held: bool) -> _PendingExtraction:
    """Extract every NAMED pending session into ``loop``'s cumulative graph.

    The single extraction stage shared by the interim tick
    (:func:`_extract_and_start_training`) and the full cycle's
    consume-pending pre-stage (inside :func:`_run_full_consolidation_sync`,
    which runs at ``max_interim_count == 0``, where no interim slot is ever
    minted so pending sessions must be consumed by the full fold itself).

    Sessions are filtered to ``SessionClass.NAMED``: the tick already dropped
    UNIDENTIFIABLE sessions and retired expired HOLDABLE ones; this filters the
    executor-time snapshot so extraction never runs on an unattributed session.

    Voice eviction fires when the pending batch contains ANY document session.
    Document chunks have no density bound — a dense ~934-word chunk (the
    current ``paramem.graph.document_chunker._DOC_MAX_TOKENS`` ceiling,
    derived from the anonymize-call token envelope — was ~1500 words before
    that derivation landed) is the regime that exhausts VRAM on this 8 GiB
    host once the ~1.5 GiB STT+TTS GPU pair sits on top of the 4-bit base
    and the extraction chain's working-set peak.  A *mixed* batch (one
    transcript probe + several dense docs) is the
    case that bit us: one transcript session used to keep voice resident through
    the dense doc extraction and the plausibility filter's KV-cache growth OOM'd
    mid-generate.  Eviction is cheap — the CPU STT/TTS pair stays resident, so
    voice still works during the cycle, just on CPU.  A pure-transcript batch
    keeps the GPU voice pair resident: turn-by-turn dialog is not the dense
    regime and likely implies recent voice activity where the lazy GPU reload
    would add latency.

    Failure handling:

    - ``VramExhausted`` is per-chunk isolation: log, record in
      ``_state["chunk_failures"]``, skip, continue on a fresh cache
      (``vram_scope``'s finally already ran ``empty_cache``).  The chunk stays
      pending.
    - ``ExtractionFailed`` aborts the WHOLE batch: the extractor actively
      refused to bake a degraded snapshot, and proceeding with the other chunks
      would silently commit a partial CV / document set.  Two origins reach
      here — the cloud ``cloud_enrich`` stage, and (since the fail-loud
      document-extraction design) a local-extraction pass
      (``local_extract``/``second_order_extract``/``procedural_extract``)
      whose output could not be parsed, detected inside
      ``loop.extract_session`` before that session's merge.  By the time the
      exception arrives here, ``loop.extract_session`` has already reset the
      merger graph, so nothing this batch extracted — including any chunk
      that merged successfully earlier in the same batch — survives in
      ``loop.merger.graph``.  ALL sessions stay pending, an
      ``extraction_failed`` incident is recorded keyed by the failing phase,
      and the abort is RECORDED in :attr:`_PendingExtraction.aborted` —
      never raised, so the caller can still see
      :attr:`_PendingExtraction.evicted_voice` and restore voice itself.
      The session loop ``break``s rather than returning, so every path
      reaches the single
      :meth:`~paramem.training.consolidation.ConsolidationLoop.take_pending_relations`
      take below exactly once — the extraction graph's lifetime ends here
      on every exit, not only the clean one.

    Args:
        loop: The process-lifetime ``ConsolidationLoop``.  ``extract_session``
            merges each session graph into its cumulative graph; this
            function's own single return takes it
            (:meth:`~paramem.training.consolidation.ConsolidationLoop.take_pending_relations`)
            into :attr:`_PendingExtraction.pending`, the value every caller
            threads into its own fold call.
        lock_held: ``True`` when the caller already holds ``gpu_lock_sync()``
            (the consume-pending pre-stage runs on the BackgroundTrainer worker
            thread, which holds it for the whole fold — a second acquisition on
            the non-reentrant lock would deadlock).  ``False`` when the caller
            does not (the interim tick runs in an executor thread), in which
            case this function acquires the lock for the extraction and RELEASES
            it before returning — so the caller's voice restore then runs
            OUTSIDE the lock (``lock_held=False``).  The eviction itself always
            happens inside the lock (``lock_held=True``).
    """
    from paramem.server.consolidation import SessionClass, classify_session
    from paramem.server.gpu_lock import gpu_lock_sync

    config = _state["config"]
    session_buffer = _state["session_buffer"]
    speaker_store = _state.get("speaker_store")

    def _is_anon(sid: "str | None") -> bool:
        return bool(speaker_store is not None and sid and speaker_store.is_anonymous(sid))

    named_ids: set[str] = {
        f["session_id"]
        for f in session_buffer.pending_facts()
        if classify_session(
            speaker_id=f["speaker_id"],
            is_anonymous=_is_anon(f["speaker_id"]),
            has_voice_embedding=f["has_voice_embedding"],
        )
        == SessionClass.NAMED
    }
    pending_sessions = [s for s in session_buffer.get_pending() if s["session_id"] in named_ids]

    result = _PendingExtraction(
        episodic_rels=[],
        procedural_rels=[],
        session_ids=[],
        failed_session_ids=set(),
        speaker_ids=[],
        evicted_voice=bool(pending_sessions)
        and any(s.get("source_type") == "document" for s in pending_sessions),
        pending=PendingRelations(episodic=[], procedural=[]),
        per_session=[],
        chunk_failures=[],
        enrichment_signals=[],
        vram_headroom_warning=None,
    )
    _headroom_sink: dict = {}

    with nullcontext() if lock_held else gpu_lock_sync():
        if result.evicted_voice:
            _set_voice_pipeline_profile("cpu", lock_held=True)

        for session in pending_sessions:
            session_id = session["session_id"]
            session_speaker_id = session.get("speaker_id")
            source_type = session.get("source_type", "transcript")
            result.session_ids.append(session_id)

            def _session_record(
                status: str, *, episodic_count: int = 0, procedural_count: int = 0
            ) -> dict:
                """This iteration's ``per_session`` row — the one place
                that shape is built, for every outcome (vram_exhausted,
                extraction_failed, ok) this session can reach."""
                return {
                    "session_id": session_id,
                    "speaker_id": session_speaker_id,
                    "source_type": source_type,
                    "episodic_count": episodic_count,
                    "procedural_count": procedural_count,
                    "status": status,
                }

            if loop.shutdown_requested:
                logger.info("Shutdown — stopping extraction early")
                break

            speaker_name = None
            if speaker_store is not None:
                speaker_name = speaker_store.resolve_speaker_name(session_speaker_id)

            try:
                # Pre-chunk headroom check: warn (no abort) when free VRAM has
                # dropped below the configured KV-cache buffer.  vram_scope
                # catches an actual OOM mid-generate and re-raises it as
                # VramExhausted (so /status shows the failure without log
                # scraping); this is the early operator signal that the booked
                # headroom is being consumed.  The sink is this run's own
                # dict, never _state directly (a non-staging run must not
                # mutate production attention state) — a staging caller
                # copies it into _state["vram_low_headroom_warning"] itself.
                check_vram_headroom(
                    session_id,
                    int(config.vram.vram_cache_headroom_gib * 2**30),
                    _headroom_sink,
                )
                with vram_scope(session_id):
                    episodic_rels, procedural_rels = loop.extract_session(
                        session["transcript"],
                        session_id,
                        speaker_id=session_speaker_id,
                        speaker_name=speaker_name,
                        enrichment_provider=config.consolidation.extraction_enrichment_provider,
                        enrichment_provider_model=config.consolidation.extraction_enrichment_provider_model,
                        enrichment_provider_endpoint=config.consolidation.extraction_enrichment_provider_endpoint
                        or None,
                        plausibility_judge=config.consolidation.extraction_plausibility_judge,
                        plausibility_stage=config.consolidation.extraction_plausibility_stage,
                        source_type=source_type,
                        event_time=session["started_at"],
                    )
            except VramExhausted as exc:
                phase = exc.args[0] if exc.args else "unknown"
                logger.warning(
                    "Chunk %s OOM at phase=%s — skipping; remains pending for retry",
                    session_id,
                    phase,
                )
                result.failed_session_ids.add(session_id)
                result.chunk_failures.append(
                    {
                        "session_id": session_id,
                        "phase": phase,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                result.per_session.append(_session_record("vram_exhausted"))
                continue
            except ExtractionFailed as exc:
                logger.error(
                    "Cycle aborted: chunk %s extraction failed at phase=%s — %s. "
                    "All %d session(s) in this batch remain pending; next cycle will retry.",
                    session_id,
                    exc.phase,
                    exc.reason,
                    len(result.session_ids),
                )
                result.chunk_failures.append(
                    {
                        "session_id": session_id,
                        "phase": exc.phase,
                        "reason": exc.reason,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                # extraction_failed is unconditional for every run — a hard
                # input fault the operator must see regardless of who asked.
                record_incident(
                    data_state_dir(config.paths.data),
                    type="extraction_failed",
                    key=str(exc.phase),
                    severity="failed",
                    summary=f"Consolidation: extraction failed at phase {exc.phase}",
                    detail={
                        "type": "extraction_failed",
                        "phase": exc.phase,
                        "reason": exc.reason,
                        "session_id": session_id,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                result.per_session.append(_session_record("extraction_failed"))
                result.aborted = exc
                break

            for rel in episodic_rels:
                rel["speaker_id"] = session_speaker_id
                # Stamp the real session id so provenance survives the
                # GraphMerger union into edge["sessions"].  Co-located with
                # the speaker_id stamp.
                rel["session_id"] = session_id
            for rel in procedural_rels:
                rel["speaker_id"] = session_speaker_id
                rel["session_id"] = session_id

            result.episodic_rels.extend(episodic_rels)
            result.procedural_rels.extend(procedural_rels)
            result.speaker_ids.append(session_speaker_id)
            result.per_session.append(
                _session_record(
                    "ok",
                    episodic_count=len(episodic_rels),
                    procedural_count=len(procedural_rels),
                )
            )

            # This session's own enrichment signal — no incident is
            # written here (see arbitrate_enrichment_incidents's own
            # docstring for why the write moved to the staging caller).
            result.enrichment_signals.append(enrichment_signal(loop, session_id))

        # The single door out of the extraction accumulation: captures the
        # merged product (episodic/procedural, already split) and resets the
        # keying surface, on every exit path — clean, OOM-skipped, or
        # ExtractionFailed abort alike.
        result.pending = loop.take_pending_relations()

    result.vram_headroom_warning = _headroom_sink.get("vram_low_headroom_warning")
    return result


def _extract_and_start_training():
    """Extract pending sessions and submit a single interim-training job.

    Runs in executor thread.  Extraction runs first: it holds the GPU lock and
    produces the per-batch (episodic_rels, procedural_rels) tuple.  Training
    then submits one callable to ``BackgroundTrainer`` that trains into the
    ``episodic_interim_<stamp>`` slot for the CURRENT consolidation window
    (``stamp`` is the refresh-cadence window floor from
    ``current_interim_stamp`` — e.g. the 12:00 window stamps
    ``episodic_interim_20260706T1200``).  When a slot for the current
    window already exists, that slot is retrained in place; a new
    stamp/slot is minted only when a new window opens.  Training is
    new-plus-replay: the slot is retrained on its existing keys reconstructed
    from adapter weights UNION the new batch, not the new batch alone.
    ``max_interim_count`` is the overflow ceiling governing what happens
    when a NEW window's mint would exceed it — an overflow slot beyond the
    cap (when ``interim_overflow_slack > 0``) or ``cap_pending`` (sessions
    stay pending for the next tick), never absorption into an existing
    slot.  Main adapters are only updated by the full-cycle path that
    calls ``loop.consolidate(...)``.
    """
    config = _state["config"]
    session_buffer = _state["session_buffer"]

    # Create or reuse consolidation loop — needed here for the extraction pass,
    # earlier than _run_stage_b_cycle's own get-or-create at training time below.
    loop = get_or_create_consolidation_loop(_state)

    # --- Extract all sessions ---
    # ``lock_held=False``: this runs in an executor thread that holds no GPU
    # lock, so the extraction stage acquires ``gpu_lock_sync()`` itself and
    # releases it before returning.  Every voice restore below therefore runs
    # OUTSIDE the lock (``lock_held=False``) — including the abort restore.
    extraction = _extract_pending_sessions(loop, lock_held=False)
    all_episodic_rels = extraction.episodic_rels
    all_procedural_rels = extraction.procedural_rels
    session_ids = extraction.session_ids
    failed_session_ids = extraction.failed_session_ids

    # Staging-only bookkeeping the extraction stage itself no longer
    # performs: this is an INTERIM dispatch, so the batch's own OOM-skip
    # records, enrichment incidents, and low-headroom attention row are
    # this run's to adopt.
    _state.setdefault("chunk_failures", []).extend(extraction.chunk_failures)
    loop.arbitrate_enrichment_incidents(extraction.enrichment_signals)
    if extraction.vram_headroom_warning is not None:
        _state["vram_low_headroom_warning"] = extraction.vram_headroom_warning

    if extraction.aborted is not None:
        # Whole-batch abort (ExtractionFailed): every session stays pending and
        # the next tick retries the batch.  The stage already logged it and
        # recorded the incident; reclaim the voice pipeline it evicted (the
        # cycle drops out without reaching any finalize) and clear the flag so
        # the retry path is not blocked by "deferred_already_running".
        _end_voice_eviction(lock_held=False)
        _consolidation_terminal(None)
        return

    # No-facts fast path: a zero-relation batch never reaches
    # run_consolidation_cycle at all, so it never dispatches to the BG
    # trainer and never acquires the GPU lock for a cycle that will train
    # nothing — the same outcome stage_event's own no-facts exit
    # ("terminates here... no shadow tree, no ledger, nothing to dispose")
    # and run_consolidation_cycle's own noop guard implement one level
    # down, for every OTHER caller that reaches them with an already-known
    # non-empty batch. Retirement here goes through the extraction
    # tracker (extraction.completed_session_ids), never the ledger: a
    # no-facts outcome produces no ledger to retire from, so this is the
    # one and only site that can retire these sessions.
    if not all_episodic_rels and not all_procedural_rels:
        logger.info("No relations extracted — skipping")
        _retire_extracted_sessions(extraction, session_buffer, loop, config)

        _end_voice_eviction(lock_held=False)

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
            _no_facts_detail = {
                "sessions": len(session_ids),
                "skipped_oom": len(failed_session_ids),
                "episodic_rels": 0,
                "procedural_rels": 0,
            }
            try:
                record_last_run(
                    data_state_dir(_state["config"].paths.data),
                    op_type="consolidation",
                    outcome="no_facts",
                    summary=f"No facts extracted from {len(session_ids)} session(s)",
                    detail=_no_facts_detail,
                )
            except Exception:
                logger.exception("Failed to record no_facts run status (non-fatal)")

        _consolidation_terminal(_finalize_no_facts)
        return

    # --- Simulate mode: peer storage backend ---
    # Same upstream pipeline as train (extraction → dedup → key assignment →
    # contradiction handling → SimHash registry); the persistence venue is
    # graph.json under adapter_dir/<tier>/ instead of LoRA weight updates.
    # mark_consolidated runs in both branches — sessions retire when their
    # work has been persisted, regardless of medium. Inference reads from
    # the graph.json via DiskMemorySource at retrieval time.

    if config.consolidation.mode == "simulate":
        primary_speaker_sim = extraction.speaker_ids[-1] if extraction.speaker_ids else ""
        # Callsite 3: scheduled-tick simulate.  The caller does NOT hold the GPU
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
            pending=extraction.pending,
            schedule=config.consolidation.refresh_cadence,
            max_interim_count=config.consolidation.max_interim_count,
            session_ids=extraction.completed_session_ids(session_buffer),
        )
        # Per-adapter indexed_key_registry.json files carry the unified simhash
        # map; MemoryStore.read_simhash_registry_from_disk merges the "simhash"
        # key out of each one for the source factory.
        #
        # A "noop" result means no tier committed this cycle, so cycle_count
        # (which now advances only once run_build_and_publish confirms a
        # bundle all_live — see that method's own docstring) never moved for
        # it either; cycle_count is a derivation anyway — the loop's counter
        # at boot is the maximum tier_cycle across every tier's own
        # key_metadata.json — so there is no whole-store counter to flush
        # here.
        # Per-tier graph.json is written by write_tier_slot (via
        # _write_built_tier) inside run_consolidation_cycle's
        # stage_event/run_build_and_publish spine; cycle_<N>/ snapshots
        # are dropped.

        _end_voice_eviction(lock_held=False)

        # State mutations + router reload — post to the event loop so the
        # router cache and inference path see the freshly-written state
        # atomically with the consolidating flag clear.  The event-kind
        # finalizer, exactly the shape the resume path already proves
        # sufficient for both venues (_finish_resumed_event selects
        # _finalize_interim/_finalize_full by ledger.event alone, never by
        # venue) — simulate is peer storage: the ledger the simulate venue's
        # stage_event/run_build_and_publish wrote is a real pending record
        # too, and _finalize_interim's own retire-then-dispose is what keeps
        # a completed simulate tick from being burned as a resume of a
        # finished event on the next dispatch.  Every venue-derived
        # reporting field (the "simulated" outcome string, the run-status
        # summary) is read from sim_result — never a second, hand-rolled
        # finalizer body.
        _consolidation_terminal(
            functools.partial(_finalize_interim, loop, sim_result, extraction=extraction)
        )
        return

    # --- Train into the current-window interim slot via the BG trainer ---
    # The scheduled tick trains into ``episodic_interim_<stamp>``, where
    # ``stamp`` is the current consolidation window's floor
    # (current_interim_stamp).  An existing current-window slot is
    # retrained in place — new-plus-replay: reconstructed existing keys
    # UNION the new batch; a new stamp/slot is minted only when a new
    # window opens.  ``max_interim_count`` is the overflow ceiling for that
    # mint (overflow slot or cap_pending), never absorption into an
    # existing slot.  Main adapters are NOT touched here — they only
    # change at the full-cycle boundary, where the full fold
    # collapses the accumulated interim slots into episodic / semantic /
    # procedural.  Freshness-wins router order: probing the newest interim
    # before main means recently
    # learned facts surface ahead of the stale main snapshot.
    primary_speaker = extraction.speaker_ids[-1] if extraction.speaker_ids else ""
    schedule = config.consolidation.refresh_cadence
    max_interim_count = config.consolidation.max_interim_count
    interim_overflow_slack = config.consolidation.interim_overflow_slack

    def _run_interim_training(loop, bt) -> "tuple[str, Callable[[], None] | None]":
        """Execute the interim training pass + post-train bookkeeping.

        Runs on the BG trainer worker thread under the GPU lock (the entry
        cooldown gate, the try/except crash envelope, and the model-handle
        refresh are owned by ``_run_stage_b_cycle``).  ``run_consolidation_cycle``
        already stages, writes, and publishes the interim slot + registry +
        key metadata durably inside the fold; this closure runs only the
        cross-cycle bookkeeping the fold does not own (ring-cap incidents,
        session pinning, and — via the returned finalizer — session marking,
        router reload, state updates) after each cycle.
        """
        # Callsite 4: BG-worker interim train.  Runs inside the BG worker
        # thread under ``gpu_lock_sync()`` (acquired by
        # ``_run_callable_queue``), driving the unified
        # ``run_consolidation_cycle``.  Fire-and-forget semantics are
        # preserved — this is NOT converted to ``_await_bg_cycle``.
        result = loop.run_consolidation_cycle(
            all_episodic_rels,
            all_procedural_rels,
            speaker_id=primary_speaker,
            mode="train",
            run_label=f"tick-{primary_speaker or 'anon'}",
            pending=extraction.pending,
            schedule=schedule,
            max_interim_count=max_interim_count,
            interim_overflow_slack=interim_overflow_slack,
            session_ids=extraction.completed_session_ids(session_buffer),
        )

        logger.info(
            "Scheduled-tick interim training: mode=%s, adapter=%s, new_keys=%d",
            result.get("mode"),
            result.get("adapter_name"),
            len(result.get("new_keys", [])),
        )

        # Keep sessions pending when their facts were NOT successfully
        # encoded. Holding a session pending for a non-encoding outcome is
        # structural, not a mechanism: a recall-gate rejection raises
        # ``RecallGateRejected`` out of ``run_consolidation_cycle`` and
        # never reaches this success path at all (see the crash path
        # below) — *session_ids* still names every contributing session,
        # but nothing here calls ``mark_consolidated`` for them, so they
        # stay pending on the crash path exactly like ABORT/CAP_PENDING.
        # The two outcomes that DO reach here with sessions to pin are
        # ABORT (training yielded to an inference request) and CAP_PENDING
        # (the interim ring was full) — neither made an encoding attempt,
        # so the sessions they touched are pinned (kept pending) below
        # rather than retired.
        _cycle_mode = result.get("mode", "trained")

        # The no-staging terminal's own retirement fallback (no ledger was
        # ever written, so _finalize_interim's ledger-based retirement has
        # nothing to read) runs inside _finalize_interim itself now, fed by
        # the *extraction* passed to the finalizer below — one call site,
        # shared by every venue that can reach this terminal, rather than a
        # copy hand-rolled at each dispatch site.

        # --- Loud incidents for ring-cap states ---
        # Both are deduped by the oldest-interim stamp so one incident fires per
        # stuck cycle and reopens only when a new cycle's interims become oldest.
        _interim_incident_state_dir = data_state_dir(_state["config"].paths.data)
        _overflow_inc = _overflow_incident_for(_cycle_mode, result.get("overflow_slot", False))
        if _overflow_inc is not None:
            _inc_type, _inc_severity = _overflow_inc
            _inc_key = _oldest_interim_stamp(config) or "unknown"
            _inc_summaries = {
                "interim_overflow_pending": (
                    "Interim ring and overflow both full — sessions kept pending "
                    "until the full fold drains the ring"
                ),
                "interim_cap_reached": (
                    "Interim ring full — overflow slot minted; full fold is overdue"
                ),
            }
            try:
                record_incident(
                    _interim_incident_state_dir,
                    type=_inc_type,
                    key=_inc_key,
                    severity=_inc_severity,
                    summary=_inc_summaries[_inc_type],
                    detail={
                        "oldest_interim_stamp": _inc_key,
                        "type": _inc_type,
                    },
                )
            except Exception:
                logger.exception("_run_interim_training: failed to record %s incident", _inc_type)

        # Pin every session this cycle touched (keep pending) on a
        # non-encoding outcome (ABORT / CAP_PENDING): none of them made an
        # encoding attempt, so none may be retired below.
        if _cycle_mode in _INTERIM_NON_ENCODING_OUTCOMES:
            failed_session_ids.update(session_ids)

        # Disk I/O — safe from any thread.  Key-metadata persistence is now
        # written durably inside the fold itself, ahead of the interim
        # tier's own registry / simhash commit signal (see
        # paramem.memory.persistence.publish_tier_registry).
        # Promotion runs at the full fold rather than here, so the key is
        # still in episodic when its adapter weights are probed during
        # reconstruction.
        #
        # Session retirement itself does NOT happen here: this cycle's
        # interim slot is durably committed inside run_consolidation_cycle
        # above (stage_event's shadow tree, written and published by
        # run_build_and_publish), but retire-then-dispose ordering makes the
        # finalizer (_finalize_interim, via _retire_ledger_sessions_and_dispose)
        # the sole retirement site — it re-reads the ledger fresh rather than
        # trusting a value captured earlier in this process, so a crash
        # between this point and the finalizer resumes to complete retirement
        # exactly once instead of retiring twice from two different sources.

        # Voice restore is not this closure's job: _run_stage_b_cycle's own
        # worker wraps this whole body(loop, bt) call in
        # _end_voice_eviction(lock_held=True), covering the success terminal
        # and the crash terminal alike from one call site.

        finalizer = functools.partial(_finalize_interim, loop, result, extraction=extraction)
        return _cycle_mode, finalizer

    _run_stage_b_cycle(
        kind="training_crash",
        incident_key="interim",
        failure_summary=f"Interim training crashed — {len(session_ids)} session(s) still pending",
        failure_detail={"sessions": len(session_ids)},
        body=_run_interim_training,
    )
    logger.info(
        "Extraction done — interim-training job submitted to BG trainer (%d sessions)",
        len(session_ids),
    )


def _finalize_full_status_only(
    *, outcome: str, summary: str, detail: dict, touch_last_consolidation: bool = False
) -> None:
    """Record-only finalizer for full-cycle terminals that touch no store/router
    state: ``aborted`` and ``noop``.

    Each of these two terminals already ran its own outcome-specific
    logging inside the full-cycle body before returning; this finalizer's
    sole job is the durable run-status record and the flag clear, both of
    which must happen atomically with everything else ``_consolidation_terminal``
    guards.

    Args:
        outcome: Run-status outcome label (also used in the exception log
            on a record-write failure).
        summary: Human-readable run-status summary.
        detail: Run-status detail payload.
        touch_last_consolidation: When ``True`` (the ``noop`` terminal only —
            matches pre-refactor behavior; ``aborted`` never stamped this
            field), updates ``_state["last_consolidation"]`` before
            recording.  Distinct from the on-disk per-tier window stamp.
    """
    if touch_last_consolidation:
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    try:
        record_last_run(
            data_state_dir(_state["config"].paths.data),
            op_type="consolidation",
            outcome=outcome,
            summary=summary,
            detail=detail,
        )
    except Exception:
        logger.exception("Failed to record %s run status (non-fatal)", outcome)


def _finalize_full(
    loop,
    result: dict,
) -> None:
    """CPU-only event-loop closure for the full-cycle ``full_trained`` terminal.

    The store is already fully published by this point — ``run_build_and_publish``'s
    ``adopt_increments`` converged it (registry, rows, entries, per tier) inside
    the same locked act that took the bundle live, and its own per-bundle
    ``router.reload()`` already ran for this event.  This finalizer's job is
    everything outside that: manifest revalidation, the unverified-tier
    incident sweep, retiring what the ledger recorded, disposing the event's
    record, and the durable run-status row.

    Step order is load-bearing:

    a. ``_revalidate_adapter_manifests`` — reads fresh on-disk slots and
       prunes stale interim ``adapter_manifest_status`` rows.
    b. The unverified-tier incident sweep — the SOLE operator-visible
       reporter for an unverified tier (the row-driven attention populator
       deliberately stays silent for it), over ``result["tier_bindings"]``
       — the publish verdict ``run_build_and_publish`` already computed
       for every tier this event's ledger names (one ``verify_tier_binding``
       read each, populated only when the event reached ``all_live``),
       never a fresh whole-tree ``verify_adapter_tree`` walk.  Protected: a
       fault here must be logged, never allowed to wedge the finalizer
       before ``_state["consolidating"]`` clears.
    c. Retire the ledger's own recorded sessions (completed extractions
       only, re-read from the ledger itself — never a value captured
       earlier in the process) and dispose the event's record — not gated
       on the consume-pending pre-stage having run, since the ledger is
       the source of truth for what to retire either way.
    d. ``_state`` flags / result bookkeeping — ``last_consolidation``,
       ``tier_weight_state`` (:func:`_record_tier_weight_state`, the go-live
       promote path's write site — a tier may just have gone from cold to
       trained-warm) etc.
       ``full_consolidation_overdue`` resolves only when every tier the
       event's ledger names went live (``result["completed"]``): a
       partially completed event (one whose bundle has not all gone live)
       must not resolve it.  This finalizer is reachable both from a fresh
       dispatch (where reaching it already implies ``result["completed"]``
       is ``True`` — the caller routes an aborted/no-op fresh result to
       ``_finalize_full_status_only`` instead) and from a resumed event
       (:func:`_run_pending_event_resume`, which calls this finalizer
       unconditionally on whatever :func:`_finish_resumed_event` returns).
       A resumed result whose bundle yielded mid-publish
       (``result["aborted"]`` or an otherwise incomplete ``result["completed"]``)
       therefore reports outcome ``"aborted"`` rather than ``"full_trained"``,
       never stamps ``last_consolidation``, and skips resolving
       ``consolidation_crash`` / ``vram_exhausted`` / ``extraction_failed`` —
       the operator's stuck-fold signal must survive an abort, not be
       cleared by one.

    Args:
        loop: The cycle's ``ConsolidationLoop`` (post-fold PEFT rebind).
        result: The ``loop.consolidate(...)`` (or, on a resumed event,
            :func:`_finish_resumed_event`) return dict.  ``result["aborted"]``
            and ``result["completed"]`` together decide the recorded
            outcome and which incidents may auto-resolve — see step d above.
    """
    # a. Re-validate manifests now that main slots have been re-saved
    #    with a fresh registry hash + window_stamp.  Without this,
    #    /status keeps showing "FINGERPRINT MISMATCH" until restart.
    _revalidate_adapter_manifests(_state)
    # b. Protected: this is the SOLE operator-visible reporter for an
    # unverified tier (attention.py's row-driven populator deliberately
    # stays silent for it) — a fault here must be logged, never allowed
    # to wedge the finalizer before _state["consolidating"] clears below.
    _config = _state["config"]
    try:
        _record_unverified_tier_incidents(_config, result.get("tier_bindings", {}))
    except Exception:
        logger.exception(
            "Post-fold tier-incident bookkeeping failed (non-fatal); finalization continues"
        )
    # c. Retire what the ledger recorded, then dispose.  A crash between the
    # two re-enters, finds every tier entry present, retires an
    # already-retired set (idempotent) and disposes.
    _retire_ledger_sessions_and_dispose(loop, disposed=bool(result.get("completed")))
    # d. Result bookkeeping.  An aborted or otherwise incomplete result
    # (reachable only from a resumed event whose bundle yielded mid-publish —
    # the fresh dispatch site routes that shape to
    # `_finalize_full_status_only` before ever calling this finalizer)
    # reports its own "aborted" outcome, never stamps last_consolidation,
    # and skips the incident resolution a genuine success clears.
    total_keys = len(loop.store.all_active_keys())
    _full_completed = bool(result.get("completed"))
    if result.get("aborted") or not _full_completed:
        _full_outcome = "aborted"
    else:
        _full_outcome = "full_trained"
    if _full_completed:
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        # Every main tier this event's ledger names may just have gone
        # live (the go-live promote path, paramem.training.go_live.publish_bundle,
        # mounted its written slot and promoted it onto ctx.model — the SAME
        # object as loop.model / _state["model"], fixed at load time) —
        # re-snapshot the weight-state record here rather than leaving
        # /status to measure it.
        _record_tier_weight_state(_state, loop.model, _config)
    _full_detail = {
        "tiers_rebuilt": result.get("tiers_rebuilt", []),
        "total_keys": total_keys,
    }
    try:
        record_last_run(
            data_state_dir(_state["config"].paths.data),
            op_type="consolidation",
            outcome=_full_outcome,
            summary=f"Full cycle {_full_outcome}: {total_keys} total keys",
            detail=_full_detail,
        )
        # Auto-resolve op-level incidents cleared by a successful full cycle —
        # gated on completion so an aborted/incomplete result leaves the
        # operator's stuck-fold signal in place.
        _full_state_dir = data_state_dir(_state["config"].paths.data)
        if _full_completed:
            resolve_incidents_by_type(_full_state_dir, "consolidation_crash")
            resolve_incidents_by_type(_full_state_dir, "vram_exhausted")
            resolve_incidents_by_type(_full_state_dir, "extraction_failed")
        # A refuse-and-hold resume that later completes (the FOREIGN write was
        # resolved out-of-band, or the record was replaced by a disposing
        # door and this is a fresh event) clears the incident it opened.
        resolve_incidents_by_type(_full_state_dir, "consolidation_resume_blocked")
        # Every full-topology fold (a full fold or a reconcile) absorbs the
        # interim ring and drains it — overdue and ring-cap incidents
        # resolve only once the event's ledger is actually gone (every
        # planned tier verified live) — a partially completed event must
        # not report a drain that did not happen.
        if _full_completed:
            resolve_incidents_by_type(_full_state_dir, "full_consolidation_overdue")
            resolve_incidents_by_type(_full_state_dir, "interim_cap_reached")
            resolve_incidents_by_type(_full_state_dir, "interim_overflow_pending")
    except Exception:
        logger.exception("Post-full-cycle run-status/incident bookkeeping failed (non-fatal)")
    logger.info("Full cycle bookkeeping complete — %d total keys", total_keys)


def _run_full_consolidation_sync(event: "Literal['full', 'reconcile']") -> None:
    """Submit a full-topology consolidation event — an ordinary full fold or
    a reconcile — under *event*'s door name.

    A reconcile (``/reconsolidate``) IS a full consolidation whose input
    excludes pending sessions: one fold topology throughout.  Runs
    ``loop.consolidate(mode=..., event=...)`` on the BG trainer so the GPU
    lock is held for the entire per-tier rebuild (the train fold's entry
    guard requires this — calling without the lock raises).  Both venues
    fold the same input — the memory store, whose main-tier and interim-slot
    registries and entries carry every active key — and both end by
    reloading the router.  In train mode the fold additionally trains each
    main adapter on the cumulative keyed-pair set and persists the weights;
    on a failed recall-sanity check ``tier_backup_scope`` restores only the
    in-VRAM state of the one tier that was training — the tier's on-disk and
    live-serving state are untouched — and the fold aborts that tier.
    Warm start is uniform for every tier of every event — no cold-start arm:
    a resident tier's weights are kept, and the funnel's staging copy
    warm-starts from them (``paramem.training.trainer.train_adapter``). The
    tier itself is never deleted or recreated by this call — only a resident
    tier's LoRA config that no longer matches the tier config is recreated
    (still never written live) ahead of the snapshot via
    ``paramem.models.loader.ensure_adapter_matching``.  In simulate mode it
    touches no PEFT weights and persists each main tier's projection into a
    fresh written slot under ``<adapter_dir>/<tier>/`` (``graph.json``, via
    :func:`~paramem.adapters.slot.write_slot` — never a tier-root path) —
    the bound slot ``DiskMemorySource`` reads back at the next hydration.

    Whether there is anything to consolidate is decided before dispatch; the fold
    itself has no content gate.

    After a successful merge the main adapter manifests are stamped with
    the current registry hash, restoring boot-time mount-ability that
    interim cycles otherwise drift away from.

    Args:
        event: ``"full"`` or ``"reconcile"`` — the door name, bound by the
            arbitrator at the dispatch site (the executor contract carries
            no arguments, so it arrives through a ``functools.partial``) and
            recorded verbatim in the ledger head.  Both run the identical
            fold: the interim ring is always recalled, absorbed into the
            main tiers, and reaped.  Only pending sessions differ — a
            ``"full"`` event at ``max_interim_count == 0`` consumes them via
            the pre-stage below; a ``"reconcile"`` event never does, so they
            stay pending no matter what ``max_interim_count`` says.
    """
    config = _state["config"]

    def _run_full_cycle(loop, bt) -> "tuple[str, Callable[[], None] | None]":
        """Run on the BG trainer worker thread under the GPU lock.

        The entry cooldown gate, the try/except crash envelope, and the
        model-handle refresh are owned by ``_run_stage_b_cycle``.
        """
        # ------------------------------------------------------------------
        # Consume-pending pre-stage (max_interim_count == 0, ordinary full
        # folds only).
        #
        # At count == 0 no interim slots are ever minted, so pending sessions
        # must be extracted directly here, before the fold — this pre-stage's
        # own take_pending_relations() call captures the merged product as
        # `extraction.pending`, threaded into loop.consolidate(pending=...)
        # below, and trains them into the main tiers.  The standard full cycle
        # (count > 0)
        # collapses already-trained interim slots into main and runs no
        # extraction chain at all — hence no pre-stage and no voice eviction.
        # A reconcile event never consumes pending sessions by definition —
        # pending sessions stay pending — so the pre-stage does not run there
        # at any count.
        #
        # HARD CONSTRAINT: the extraction stage is called with lock_held=True.
        # This closure already runs under the BG trainer worker's GPU lock
        # (BackgroundTrainer._run_callable_queue), and any second acquisition of
        # the non-reentrant _gpu_thread_lock would deadlock the worker thread.
        # extract_session / ExtractionPipeline / GraphMerger acquire zero locks
        # and never call .submit() (grep-confirmed safe).
        #
        # Extraction-failed sessions are NOT marked consolidated; they stay
        # pending for the next tick.  A successfully-extracted session is marked
        # on BOTH fold success and fold noop (extraction succeeded but no NEW
        # facts after dedup — the session is processed, not retried).  This
        # prevents already-processed sessions from accumulating unboundedly.
        # ------------------------------------------------------------------
        _consume_pending = (
            event != "reconcile"
            and config.consolidation.max_interim_count == 0
            and config.consolidation.mode != "simulate"
        )
        _cp_session_buffer = _state["session_buffer"]
        extraction: _PendingExtraction | None = None

        if _consume_pending:
            extraction = _extract_pending_sessions(loop, lock_held=True)

            # Staging-only bookkeeping the extraction stage itself no longer
            # performs: this is a FULL/consume-pending dispatch, so the
            # batch's own OOM-skip records, enrichment incidents, and
            # low-headroom attention row are this run's to adopt.
            _state.setdefault("chunk_failures", []).extend(extraction.chunk_failures)
            loop.arbitrate_enrichment_incidents(extraction.enrichment_signals)
            if extraction.vram_headroom_warning is not None:
                _state["vram_low_headroom_warning"] = extraction.vram_headroom_warning

            # The voice restore is not this pre-stage's job: _run_stage_b_cycle's
            # own worker wraps this whole body(loop, bt) call in
            # _end_voice_eviction(lock_held=True), so the restore fires AFTER
            # the fold — the heaviest GPU phase on this 8 GiB card runs
            # without the ~1.5 GiB STT+TTS pair on top of it.

            if extraction.aborted is not None:
                # Whole-batch abort: all sessions stay pending, next tick
                # retries.  The stage logged it and recorded the incident.
                return "extraction_failed", None

        result = loop.consolidate(
            mode=config.consolidation.mode,
            event=event,
            trainer=bt,
            router=_state.get("router"),
            pending=(extraction.pending if extraction is not None else None),
            session_ids=(
                extraction.completed_session_ids(_cp_session_buffer)
                if extraction is not None
                else None
            ),
        )
        logger.info(
            "Full cycle complete — tiers_rebuilt=%s",
            result.get("tiers_rebuilt"),
        )

        # Layering boundary. ``loop.consolidate(...)`` already
        # finished its internal finalize on its way out (commit — registries
        # and payload, weights in train or per-tier graph.json in simulate —
        # then interim reap and router reload; see its internal finalize
        # block) regardless of whether anything was rebuilt: even a fold
        # that rebuilt nothing still commits its registry mutations (a
        # no-retrain restamp) before returning.  Reaping the interim slots
        # is the narrower guard — that only happens when the fold also
        # rebuilt a tier, inside the method, after the commit — so there is
        # no crash window in which the folded knowledge has no on-disk copy.
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
            if result.get("aborted"):
                # Training yielded (to inference, or a graceful shutdown)
                # mid-bundle -- distinct from the no-facts noop below.
                # Nothing was learned from the consume-pending pre-stage's
                # extracted content, so retiring these sessions here would
                # be an unrecoverable loss (transcripts gone, nothing
                # learned, /reconsolidate rebuilds from stored knowledge and
                # can never recover what was never encoded).  Every session
                # this pre-stage touched stays pending for the next tick's
                # retry -- mark_consolidated is not called at all.
                logger.info(
                    "Full cycle aborted mid-bundle — nothing rebuilt, sessions "
                    "kept pending for retry; consolidating flag cleared"
                )
                return "aborted", functools.partial(
                    _finalize_full_status_only,
                    outcome="aborted",
                    summary="Full cycle aborted — training yielded mid-bundle",
                    detail={
                        "tiers_rebuilt": [],
                    },
                    touch_last_consolidation=False,
                )
            # MF-A Site A: at count==0 the consume-pending pre-stage extracted
            # sessions into merger.graph, but the fold found nothing new to train
            # (all facts already present after dedup) — no ledger names these
            # sessions, so the ledger-based retirement site has nothing to
            # read.  Mark extraction-succeeded sessions consolidated so they
            # do not accumulate unboundedly.  Extraction-failed sessions keep
            # their pending status for retry.
            if extraction is not None and extraction.session_ids:
                try:
                    _retire_extracted_sessions(extraction, _cp_session_buffer, loop, config)
                except Exception:
                    logger.exception("consume-pending noop: mark_consolidated failed (non-fatal)")
            logger.info(
                "Full cycle no-op — nothing to rebuild, inner finalize already "
                "ran inside the fold; consolidating flag cleared"
            )
            return "noop", functools.partial(
                _finalize_full_status_only,
                outcome="noop",
                summary="Full cycle no-op — nothing to rebuild",
                detail={
                    "tiers_rebuilt": [],
                },
                touch_last_consolidation=True,
            )

        # The rebuilt main tiers were already persisted to disk inside the fold's
        # finalize (its commit act, before the interim reap). In the
        # train venue that persist is verified and stamps each main slot's
        # meta.json with a fresh window_stamp + registry_sha256, so the slots
        # remount on restart; in the simulate venue it writes the per-tier
        # graph.json projections. If the persist (or its verify) had failed, the
        # method would have raised and propagated to ``_run_stage_b_cycle``'s
        # crash envelope (consolidation_crash incident, sessions left pending) —
        # so reaching here means main is durable.

        # Key metadata is now persisted durably inside the fold itself, per
        # tier, ahead of that tier's own registry rewrite (see
        # paramem.memory.persistence.publish_tier_registry) — no app-layer
        # call needed.
        #
        # When max_interim_count > 0 (standard mode): the full consolidation run
        # folds interim-adapter content into main; it does NOT run the extraction
        # chain on pending sessions.  Pending sessions remain in the buffer and are
        # consumed by the next interim tick.  Marking them consolidated here would
        # permanently discard them without ever extracting or training on their
        # content.
        #
        # When max_interim_count == 0 (consume-pending mode): the pre-stage above
        # captured the pending sessions' extraction product as `pending`, and the
        # fold trained it into the main tiers.  MF-A Site B: mark
        # extraction-succeeded sessions consolidated now that the fold has persisted
        # their knowledge to the main tiers.  Extraction-failed sessions stay pending.

        # Session retirement itself does NOT happen here (count==0
        # consume-pending mode's own success path): retire-then-dispose
        # ordering makes _finalize_full (via _retire_ledger_sessions_and_dispose,
        # dispatched below) the sole retirement site — it re-reads the
        # ledger's own recorded session list fresh rather than trusting a
        # value captured earlier in this process, so a crash between this
        # point and the finalizer resumes to complete retirement exactly
        # once instead of retiring twice from two different sources.

        # The store is already fully published by this point:
        # run_build_and_publish's adopt_increments converged the live store
        # (registry, rows, entries) for every tier this event built, inside
        # the same locked act that took the bundle live.  No second,
        # off-store rebuild-and-swap is needed or run.
        loop.model.eval()

        return "full_trained", functools.partial(
            _finalize_full,
            loop,
            result,
        )

    _run_stage_b_cycle(
        kind="consolidation_crash",
        incident_key=event,
        failure_summary=f"{event.capitalize()} consolidation crashed unexpectedly",
        failure_detail={},
        body=_run_full_cycle,
    )
    logger.info("%s consolidation submitted to BG trainer", event.capitalize())


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


def _finalize_migration(loop, updated) -> None:
    """Success/terminal finalizer for the active-store-migration Stage-B cycle.

    Runs on the asyncio event loop via ``_consolidation_terminal``.  Records the
    durable run-status row and, on ``all_tiers_done``, clears
    ``pending_rehydration`` (restoring ``effective_mode`` to the operator's
    configured mode) and auto-resolves migration incidents.  Partial
    completion leaves ``pending_rehydration`` set so a re-trigger picks up
    the remaining tiers.

    Args:
        loop: The cycle's ``ConsolidationLoop`` (post-migration PEFT rebind).
        updated: The ``MigrationState`` returned by ``migrate``.
    """
    loop.model.eval()
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    # migrate() re-creates and retrains each tier it touches directly on
    # loop.model (create_adapter + train), not through the fold's go-live
    # promote path — re-snapshot here too, even on a partial run, since a
    # tier already migrated before a later tier's failure keeps its new
    # weight state regardless of the overall outcome.
    _record_tier_weight_state(_state, loop.model, _state["config"])
    _all_done = updated.all_tiers_done(loop.store.tiers_with_registry())
    _mig_outcome = "migration_complete" if _all_done else "migration_partial"
    _mig_detail = {
        "direction": updated.direction,
        "completed_tiers": list(updated.completed_tiers),
        "failed_tiers": dict(updated.failed_tiers),
    }
    try:
        record_last_run(
            data_state_dir(_state["config"].paths.data),
            op_type="consolidation",
            outcome=_mig_outcome,
            summary=f"Migration {_mig_outcome}: direction={updated.direction}",
            detail=_mig_detail,
        )
        if _all_done:
            # Auto-resolve migration incidents on successful completion.
            _mig_state_dir = data_state_dir(_state["config"].paths.data)
            resolve_incidents_by_type(_mig_state_dir, "migration_error")
            resolve_incidents_by_type(_mig_state_dir, "migration_phase_failed")
    except Exception:
        logger.exception("Post-migration run-status/incident bookkeeping failed (non-fatal)")
    if _all_done:
        _state["pending_rehydration"] = False
        _state["effective_mode"] = _state["config"].consolidation.mode
        logger.info(
            "Active-store migration complete; effective_mode=%s",
            _state["config"].consolidation.mode,
        )


def _run_active_store_migration_sync() -> None:
    """Execute the pending active-store migration on a worker thread.

    Triggered by ``_dispatch_consolidation`` when
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

    config = _state["config"]
    state = load_state(config.adapter_dir)
    if state is None:
        # Nothing to do — flag was set but state file is gone (already
        # completed by another caller). Clear pending and return.
        _state["pending_rehydration"] = False
        _state["effective_mode"] = config.consolidation.mode
        logger.info("Active-store migration: state file absent — clearing pending flag")
        _consolidation_terminal(None)
        return

    def _run_migration_on_worker(loop, bt) -> "tuple[str, Callable[[], None] | None]":
        """Run on the BG-trainer worker thread under the GPU lock.

        The entry cooldown gate, the try/except crash envelope, and the
        model-handle refresh are owned by ``_run_stage_b_cycle``.
        """
        updated = migrate(loop, config, state)

        logger.info(
            "Active-store migration done: direction=%s completed=%s failed=%s",
            updated.direction,
            updated.completed_tiers,
            list(updated.failed_tiers.keys()),
        )

        return "migration_done", functools.partial(_finalize_migration, loop, updated)

    _run_stage_b_cycle(
        kind="migration_error",
        incident_key="active_store",
        failure_summary="Active-store migration raised unexpectedly",
        failure_detail={},
        body=_run_migration_on_worker,
    )
    logger.info("Active-store migration submitted to BG trainer")


# --- Speaker enrollment (utterance-driven) ---
#
# When an unknown speaker talks, their voice embedding is stored alongside
# each transcript turn and a canonical speaker{N} id is allocated. The
# chat handler then invokes _run_enrollment_for_group synchronously on
# every anonymous turn so the LLM extractor and the follow-on
# store.enroll / claim_sessions / cleanup run before the response is
# rendered. The LLM is the sole filter — non-introduction turns return
# NONE and have no side effect. There is no background idle loop and
# no regex pre-filter; the user's own utterance is always the trigger.


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

    # Chronological order: prior session turns first, then the live turn
    # (extra_turns) last — get_conversation_turns returns turns in append order,
    # so the live turn (not yet appended to the buffer) belongs after them.
    all_turns: list[dict] = buffer.get_conversation_turns(conv_id) + list(extra_turns or [])

    if not all_turns:
        return None

    from paramem.server.gpu_lock import gpu_lock

    def _extract(turns: list[dict]) -> str | None:
        """The executor payload: this pass's trace scope + the extractor.

        Opened HERE rather than around the ``run_in_executor`` call because
        the executor runs on a worker thread, which does not inherit the
        caller's contextvars — a scope opened outside would not be visible
        to the phase the extractor opens inside. Name enrollment is
        structured extraction, so it runs on the base weights, never the
        training-active adapter.
        """
        with extraction_trace(), base_model_inference(model):
            name, _raw = extract_name_via_llm(turns, model, tokenizer)
        return name

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(None, _extract, all_turns)

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
        result = systemctl.run("show-environment", timeout=5)
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
        systemctl.run("unset-environment", *_HOLD_ENV_VARS, timeout=5)
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
                    _reload_reason = await loop.run_in_executor(
                        None,
                        lambda: _live_reload_base_model(lock_held=True),
                    )
                if _reload_reason is not None:
                    # Reload was declined (insufficient free VRAM) or failed
                    # and self-cleaned — the base model is NOT loaded. Do not
                    # load the STT/TTS GPU pair: that is exactly how a
                    # cloud-only server ends up squatting VRAM. Force voice
                    # back to CPU (outside the gpu_lock) and keep polling; the
                    # GPU may free on a later tick.
                    await loop.run_in_executor(None, _set_voice_pipeline_profile, "cpu")
                    logger.info(
                        "Auto-reclaim: reload deferred (reason=%s) — retrying next tick",
                        _reload_reason,
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

    No longer called from auto-reclaim (auto-reclaim now uses in-process
    reload via ``_live_reload_base_model`` + ``_set_voice_pipeline_profile``).
    Still used by :func:`gpu_acquire` as a fallback when in-process reload
    fails, and is kept defined for future emergency use. Process-level death
    is handled orthogonally by ``Restart=on-failure`` + ``RestartSec=30`` in
    the systemd unit.
    """
    logger.info("Restarting paramem-server service...")
    try:
        systemctl.spawn("restart", "paramem-server")
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
    project_root = find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2]
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
        try:
            systemctl.run("unset-environment", "PARAMEM_EXTRA_ARGS", timeout=5)
        except Exception:
            logger.exception("Failed to unset stale PARAMEM_EXTRA_ARGS on startup")

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        log_config=None,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
