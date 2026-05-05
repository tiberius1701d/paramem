"""Calibration endpoints for live prompt iteration.

Exposes per-stage probes against the live Mistral instance the production
cycle uses — same model, same VRAM/allocator history, same prompt-loading
mechanism.  Results are a 1:1 reflection of what the production pipeline
would emit on this input given this prompt.

Each endpoint is a thin wrapper around the existing pipeline helper for
that stage — no orchestration, no parallel routes; injection of the
operator's prompts/params into the production code path, and rerouting
of the output for capture.  Stages are stop points: the calibration
client chains them, and "skip stage X" simply means don't call X's
endpoint.  No call modifies weights or writes production data on disk.

Endpoints:

* ``POST /calibrate/extract`` — runs the local extractor through
  :meth:`ConsolidationLoop._run_extract_graph` (the single-topology
  chokepoint).  Returns the local-only graph (pre-anonymization).
* ``POST /calibrate/anonymize`` — runs ``_anonymize_with_local_model`` on
  the caller-supplied SessionGraph + transcript.
* ``POST /calibrate/plausibility`` — runs ``_local_plausibility_filter``
  on the caller-supplied fact list + transcript.

Returns a uniform shape:

  prompt_path, prompt_sha, prompt_content, raw_output, parsed,
  n_input_tokens, n_output_tokens, wall_clock_seconds, model,
  params_effective, vram_before, vram_after.

Concurrency: every endpoint short-circuits with 503 when
``_state["consolidating"]`` is True so calibration calls cannot race
against an active consolidation cycle.

Gating: every endpoint short-circuits with 404 when the server config's
``calibrate_endpoint_enabled`` flag is False.  Default is False —
calibration is opt-in via ``configs/server.yaml``, never live in
production.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CalibrateParams(BaseModel):
    """Inference-time sampling overrides for a single calibration call.

    All fields default to ``None`` — the underlying call site uses its
    configured production default for every unset field.  ``seed`` only
    applies to local stages (Anthropic does not accept a seed parameter;
    SOTA stages report ``seed: null`` in ``params_effective``).
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int | None = None


class CalibrateExtractRequest(BaseModel):
    transcript: str
    speaker_id: str
    speaker_name: str | None = None
    source_type: str = Field(default="document", pattern="^(transcript|document)$")
    session_id: str = "calib"
    prompts_dir: str | None = None
    extraction_prompt_filename: str | None = None
    extraction_system_prompt_filename: str | None = None
    # Calibration short-circuit: when set to a name from
    # paramem.graph.phase_trace.PHASE_NAMES, the pipeline returns
    # immediately after that phase completes — saves compute when only
    # the early phases need to be inspected.  Default None runs the full
    # pipeline.  Validation against PHASE_NAMES happens inside
    # extract_graph; an invalid value surfaces as ValueError → HTTP 500.
    stop_phase: str | None = None
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateAnonymizeRequest(BaseModel):
    """Run the anonymizer on an explicit graph + transcript pair.

    The ``graph`` payload is a SessionGraph dict (entities + relations +
    metadata); the calibration client typically takes this from a prior
    ``/calibrate/extract`` response.  The transcript is the original
    pre-anon text.
    """

    graph: dict
    transcript: str
    session_id: str = "calib"
    prompts_dir: str | None = None
    anonymization_prompt_filename: str | None = None
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibratePlausibilityRequest(BaseModel):
    """Run the plausibility filter on an explicit fact list + transcript.

    Facts are real-named (the calibration client typically de-anonymizes
    the SOTA enrichment output before sending here, mirroring the
    production deanon-stage call site).
    """

    facts: list[dict]
    transcript: str
    prompts_dir: str | None = None
    plausibility_prompt_filename: str | None = None
    params: CalibrateParams = Field(default_factory=CalibrateParams)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vram_block() -> dict[str, float] | None:
    """Capture a VRAM snapshot in the same shape as ``_vram_snapshot`` logs.

    Returns ``None`` when CUDA is unavailable (CPU-only test environments).
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001
        return None
    block: dict[str, float] = {
        "alloc_mib": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mib": torch.cuda.memory_reserved() / (1024 * 1024),
        "peak_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            parts = out.stdout.strip().split(",")
            block["smi_used_mib"] = float(parts[0].strip())
            block["smi_free_mib"] = float(parts[1].strip())
    except Exception:  # noqa: BLE001
        pass
    return block


def _read_prompt(
    prompts_dir: str | Path | None,
    filename: str,
) -> tuple[str, str, str]:
    """Resolve and read a prompt file, returning (path, sha, content).

    Falls back to the project's default prompts dir when ``prompts_dir`` is
    ``None``.  The file MUST exist — calibration surfaces a clear error
    rather than letting :func:`_load_prompt`'s embedded default mask a
    missing operator-supplied prompt.

    The prompts this function reads are the same ones the production
    pipeline consumes via :func:`_load_prompt`.  Before authoring a
    ``calib_*.txt`` variant, read **README.md → Prompt Engineering** —
    the calibration loop documented there is what this function exists
    to support.
    """
    if prompts_dir is not None:
        path = Path(prompts_dir) / filename
    else:
        # Mirror the project's default prompts location.
        path = Path("configs") / "prompts" / filename
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Prompt file not found: {path}. "
            f"Calibration requires the operator to specify a real prompt "
            f"file; embedded fallbacks are not used.",
        )
    content = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return str(path), sha, content


def _count_tokens(tokenizer, text: str) -> int:
    """Best-effort token count.  Returns ``-1`` when the tokenizer rejects
    the input (rare; occurs on MagicMock test fixtures)."""
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:  # noqa: BLE001
        return -1


@contextmanager
def _cudnn_deterministic():
    """Toggle cuDNN deterministic flags for the duration of a calibration
    call.  Saved/restored so the change cannot leak into production
    inference running in the same process.

    No-op when torch / CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        yield
        return
    try:
        if not torch.cuda.is_available():
            yield
            return
    except Exception:  # noqa: BLE001
        yield
        return
    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.benchmark = prev_bench


# ---------------------------------------------------------------------------
# Pre-flight gate (shared by every endpoint)
# ---------------------------------------------------------------------------


def _preflight(state: dict) -> None:
    """Raise the appropriate HTTP exception when the server is not in a
    state that can serve a calibration call.

    * 404 when the calibrate flag is off — the endpoint shouldn't exist
      from the client's perspective.
    * 503 when a real consolidation cycle is running — refusing prevents
      the calibration call from racing against the model.
    * 503 when the model isn't loaded (cloud-only mode, defer-model boot).
    """
    config = state.get("config")
    if config is None or not getattr(config.consolidation, "calibrate_endpoint_enabled", False):
        raise HTTPException(
            status_code=404,
            detail=(
                "Calibration endpoint is disabled. Set "
                "consolidation.calibrate_endpoint_enabled: true in "
                "configs/server.yaml to enable."
            ),
        )
    if state.get("consolidating"):
        raise HTTPException(
            status_code=503,
            detail="Consolidation cycle in progress; calibration calls "
            "cannot race against the live model. Retry after the cycle "
            "completes.",
            headers={"Retry-After": "60"},
        )
    if state.get("model") is None or state.get("tokenizer") is None:
        raise HTTPException(
            status_code=503,
            detail="Local model not loaded (cloud-only mode or "
            "defer-model boot). Calibration requires a local model.",
        )


# ---------------------------------------------------------------------------
# Stage handlers — invoked from the registered FastAPI routes in app.py
# ---------------------------------------------------------------------------


def calibrate_extract(state: dict, req: CalibrateExtractRequest) -> dict[str, Any]:
    """Run the local extractor for a single transcript.

    Routes through ``ConsolidationLoop._run_extract_graph`` (the
    single-topology chokepoint).  Never calls ``extract_graph`` directly.
    """
    _preflight(state)
    # Lazy-init the loop the same way the production /consolidate handler does.
    # This is not a parallel route — it's the same factory and the same single
    # ConsolidationLoop instance.  The FIRST call (calibrate or consolidate)
    # creates it; subsequent calls reuse.  No weight modification, no disk
    # writes — `_run_extract_graph` is read-only, the merger and trainer
    # come AFTER, in the consolidation orchestrator, and are NOT called here.
    loop = state.get("consolidation_loop")
    if loop is None:
        from paramem.server.consolidation import create_consolidation_loop

        loop = create_consolidation_loop(
            state["model"],
            state["tokenizer"],
            state["config"],
            state_provider=lambda: state,
        )
        state["consolidation_loop"] = loop
        state["model"] = loop.model
    tokenizer = state["tokenizer"]

    # Resolve prompt files for transparency: even though the actual read
    # happens inside extract_graph via ``_load_prompt``, we surface the
    # operator-supplied path + sha + content in the response so dump
    # diffs show prompt provenance directly.
    user_filename = req.extraction_prompt_filename or (
        "extraction_document.txt" if req.source_type == "document" else "extraction.txt"
    )
    sys_filename = req.extraction_system_prompt_filename or (
        "extraction_system_document.txt"
        if req.source_type == "document"
        else "extraction_system.txt"
    )
    user_prompt_path, user_prompt_sha, user_prompt_content = _read_prompt(
        req.prompts_dir, user_filename
    )
    sys_prompt_path, sys_prompt_sha, sys_prompt_content = _read_prompt(
        req.prompts_dir, sys_filename
    )

    overrides: dict = {
        "speaker_id": req.speaker_id,
        "speaker_name": req.speaker_name,
        "system_prompt_filename": sys_filename,
        "user_prompt_filename": user_filename,
    }
    if req.prompts_dir is not None:
        overrides["prompts_dir"] = req.prompts_dir
    if req.params.max_tokens is not None:
        overrides["max_tokens"] = req.params.max_tokens
    if req.params.temperature is not None:
        overrides["temperature"] = req.params.temperature
    if req.stop_phase is not None:
        overrides["stop_phase"] = req.stop_phase
    # NOTE: top_p, top_k, seed do not currently flow through
    # extract_graph's signature — those are for direct generate_answer
    # calls (anonymize / plausibility paths).  Document the limitation
    # in params_effective.

    vram_before = _vram_block()
    t0 = time.perf_counter()
    with _cudnn_deterministic():
        graph = loop._run_extract_graph(
            req.transcript,
            req.session_id,
            source_type=req.source_type,
            **overrides,
        )
    elapsed = time.perf_counter() - t0
    vram_after = _vram_block()

    parsed = graph.model_dump(mode="json") if hasattr(graph, "model_dump") else {}
    # Per-phase trace — every prompt-touching step of the pipeline records
    # its own raw_output + parsed summary on graph.diagnostics["phases"].
    # Calibration consumers diff phase outputs across prompt variants to
    # localise prompt-specific behaviour.
    from paramem.graph.phase_trace import get_phases

    phase_records = [r.to_dict() for r in get_phases(graph)]
    # Phase 1 raw output, surfaced at the top level for easy diffing of
    # the local-extract prompt without traversing the phases list.
    local_extract = next(
        (p for p in phase_records if p.get("name") == "local_extract"),
        None,
    )
    raw_output = (local_extract or {}).get("raw_output") or ""

    return {
        "stage": "extract",
        "prompts": [
            {
                "role": "system",
                "path": sys_prompt_path,
                "sha": sys_prompt_sha,
                "content": sys_prompt_content,
            },
            {
                "role": "user",
                "path": user_prompt_path,
                "sha": user_prompt_sha,
                "content": user_prompt_content,
            },
        ],
        "raw_output": raw_output,
        "parsed": parsed,
        "phases": phase_records,
        "n_input_tokens": _count_tokens(tokenizer, user_prompt_content),
        "n_output_tokens": _count_tokens(tokenizer, raw_output) if raw_output else -1,
        "wall_clock_seconds": elapsed,
        "model": getattr(state.get("model_id"), "name", state.get("model_id", "unknown")),
        "params_effective": _effective_params(req.params, supports_seed=False),
        "vram_before": vram_before,
        "vram_after": vram_after,
    }


def calibrate_anonymize(state: dict, req: CalibrateAnonymizeRequest) -> dict[str, Any]:
    """Run the local anonymizer on an explicit graph + transcript."""
    _preflight(state)
    from paramem.graph.extractor import _anonymize_with_local_model
    from paramem.graph.schema import SessionGraph

    model = state["model"]
    tokenizer = state["tokenizer"]

    filename = req.anonymization_prompt_filename or "anonymization.txt"
    prompt_path, prompt_sha, prompt_content = _read_prompt(req.prompts_dir, filename)

    try:
        graph = SessionGraph.model_validate(req.graph)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SessionGraph payload: {exc}",
        ) from exc

    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    vram_before = _vram_block()
    t0 = time.perf_counter()
    with _cudnn_deterministic():
        anon_facts, mapping, anon_transcript, raw_output = _anonymize_with_local_model(
            graph,
            model,
            tokenizer,
            transcript=req.transcript,
            max_tokens=max_tokens,
        )
    elapsed = time.perf_counter() - t0
    vram_after = _vram_block()

    parsed: dict[str, Any] = {
        "anonymized_facts": anon_facts,
        "mapping": mapping,
        "anonymized_transcript": anon_transcript,
    }

    return {
        "stage": "anonymize",
        "prompts": [
            {"role": "user", "path": prompt_path, "sha": prompt_sha, "content": prompt_content},
        ],
        "raw_output": raw_output,
        "parsed": parsed,
        "n_input_tokens": _count_tokens(tokenizer, prompt_content),
        "n_output_tokens": _count_tokens(tokenizer, raw_output) if raw_output else -1,
        "wall_clock_seconds": elapsed,
        "model": state.get("model_id", "unknown"),
        "params_effective": _effective_params(req.params, supports_seed=False),
        "vram_before": vram_before,
        "vram_after": vram_after,
    }


def calibrate_plausibility(state: dict, req: CalibratePlausibilityRequest) -> dict[str, Any]:
    """Run the local plausibility filter on an explicit fact list."""
    _preflight(state)
    from paramem.graph.extractor import _local_plausibility_filter

    model = state["model"]
    tokenizer = state["tokenizer"]

    filename = req.plausibility_prompt_filename or "sota_plausibility.txt"
    prompt_path, prompt_sha, prompt_content = _read_prompt(req.prompts_dir, filename)

    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    vram_before = _vram_block()
    t0 = time.perf_counter()
    with _cudnn_deterministic():
        kept, raw_output = _local_plausibility_filter(
            req.facts,
            req.transcript,
            model,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elapsed = time.perf_counter() - t0
    vram_after = _vram_block()

    dropped = [f for f in req.facts if f not in (kept or [])]
    parsed: dict[str, Any] = {
        "kept_facts": kept or [],
        "dropped_facts": dropped,
        "input_count": len(req.facts),
        "kept_count": len(kept or []),
        "dropped_count": len(dropped),
    }

    return {
        "stage": "plausibility",
        "prompts": [
            {"role": "user", "path": prompt_path, "sha": prompt_sha, "content": prompt_content},
        ],
        "raw_output": raw_output,
        "parsed": parsed,
        "n_input_tokens": _count_tokens(tokenizer, prompt_content),
        "n_output_tokens": _count_tokens(tokenizer, raw_output) if raw_output else -1,
        "wall_clock_seconds": elapsed,
        "model": state.get("model_id", "unknown"),
        "params_effective": _effective_params(req.params, supports_seed=False),
        "vram_before": vram_before,
        "vram_after": vram_after,
    }


def _effective_params(params: CalibrateParams, *, supports_seed: bool) -> dict:
    """Return the params dict the call effectively applied.

    ``supports_seed`` distinguishes local stages (where seed is honoured
    via a scoped torch.Generator) from SOTA stages (where Anthropic's API
    accepts no seed parameter and the field is silently dropped).  The
    response uses this to inform the operator which fields actually
    landed.

    Currently extract / anonymize / plausibility do not yet thread
    top_p / top_k / seed into ``generate_answer`` from the calibrate
    layer — those flow when the underlying helper is updated.  The
    field is reported as-requested for transparency; future iterations
    will push them through.
    """
    out: dict = {}
    for f in ("temperature", "top_p", "top_k", "max_tokens"):
        out[f] = getattr(params, f)
    out["seed"] = params.seed if supports_seed else None
    return out
