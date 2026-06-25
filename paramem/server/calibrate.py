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
  :class:`paramem.graph.extraction_pipeline.ExtractionPipeline` (the
  single-topology chokepoint).  The pipeline instance is held by the
  process-wide ``ConsolidationLoop`` (lazy-built on first /consolidate
  or /calibrate call); calibrate reaches it via ``loop.extraction`` so
  every flag the production cycle applies is applied here too.  Returns
  the local-only graph (pre-anonymization).
* ``POST /calibrate/anonymize`` — runs ``anonymize_with_local_model`` on
  the caller-supplied SessionGraph + transcript.
* ``POST /calibrate/plausibility`` — runs ``local_plausibility_filter``
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

from paramem.server.gpu_lock import gpu_lock_sync

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CalibrateParams(BaseModel):
    """Inference-time sampling overrides for a single calibration call.

    All fields default to ``None`` — the underlying call site uses its
    configured production default for every unset field.  ``seed`` only
    applies to local stages (Anthropic does not accept a seed parameter;
    SOTA stages report ``seed: null`` in ``params_effective``).  seed
    only affects output at temperature>0; at the default greedy
    temperature 0.0 it is a no-op.
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


class CalibrateNormalizeRequest(BaseModel):
    """Run the two-stage synonym-predicate dedup on an explicit relation list
    or a graph snapshot.

    Exactly one of ``relations`` or ``snapshot_path`` must be provided;
    supplying neither or both raises HTTP 400.

    * ``relations`` — flat list of relation dicts (each with at minimum
      ``subject``, ``predicate``, ``object`` keys), supplied directly by
      the caller.
    * ``snapshot_path`` — path to a NetworkX node-link
      ``graph_merged_snapshot.json`` on the server filesystem.  Edges are
      flattened to ``{subject, predicate, object}`` dicts; edges missing a
      ``predicate`` key are skipped.

    ``filter_prompt_filename`` and ``merge_prompt_filename`` default to
    ``graph_dedup_filter.txt`` and ``graph_dedup_merge.txt`` respectively.
    ``prompts_dir`` defaults to the project's ``configs/prompts/`` directory.
    """

    relations: list[dict] | None = None
    snapshot_path: str | None = None
    filter_prompt_filename: str | None = None
    merge_prompt_filename: str | None = None
    prompts_dir: str | None = None
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateNameRequest(BaseModel):
    """Run the name-extraction LLM on an explicit turn list.

    ``turns`` is a list of ``{"role": str, "text": str}`` dicts — the same
    shape the production enrollment path receives from
    ``_run_enrollment_for_speaker``.  When ``user_turns_only`` is ``True``
    (default, mirrors production), only ``role == "user"`` turns are fed to
    the model; assistant turns are silently excluded so salutations like
    "Good evening, user" cannot be mis-classified as name introductions.

    ``name_prompt_filename`` and ``name_system_prompt_filename`` default to
    the production prompt files.  Override to point at a scratch copy for
    live prompt iteration.
    """

    turns: list[dict]
    prompts_dir: str | None = None
    name_prompt_filename: str | None = None
    name_system_prompt_filename: str | None = None
    user_turns_only: bool = True
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
    pipeline consumes via :func:`_load_prompt`.  Prompts are external
    config — edit the files under ``configs/prompts/`` to tune; no code
    changes are needed.
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
# Shared measurement primitives (Seam A and Seam B)
# ---------------------------------------------------------------------------


class _Measurement:
    """Timing and VRAM snapshot captured around a single calibration call."""

    __slots__ = ("vram_before", "vram_after", "elapsed")

    def __init__(self) -> None:
        self.vram_before: dict | None = None
        self.vram_after: dict | None = None
        self.elapsed: float = 0.0


@contextmanager
def _measured_local_call():
    """Context manager that wraps the GPU lock and timing for every local stage.

    Captures ``vram_before``, acquires ``gpu_lock_sync()`` and
    ``_cudnn_deterministic()``, then records ``elapsed`` and ``vram_after``
    on exit.  All five calibration handlers must use this wrapper so the
    "every local stage takes the GPU lock" invariant is enforced in a single
    place.

    Yields a :class:`_Measurement` object whose attributes are populated on
    context exit.  Usage::

        with _measured_local_call() as m:
            result = do_gpu_work(...)
        # m.elapsed, m.vram_before, m.vram_after are now set.
    """
    m = _Measurement()
    m.vram_before = _vram_block()
    t0 = time.perf_counter()
    with gpu_lock_sync(), _cudnn_deterministic():
        yield m
    m.elapsed = time.perf_counter() - t0
    m.vram_after = _vram_block()


def _build_calibrate_response(
    *,
    stage: str,
    prompts: list[dict],
    raw_output: Any,
    parsed: dict,
    input_prompt_text: str,
    measurement: _Measurement,
    params: CalibrateParams,
    state: dict,
    supports_seed: bool = True,
    raw_output_for_tokens: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the uniform 9-key tail shared by every calibration stage.

    Each handler builds only the per-stage parts (``prompts`` list,
    ``raw_output``, ``parsed``) and delegates the common envelope to this
    function.  The per-stage parts are legitimately different and must NOT
    be pushed into this builder.

    The ``n_output_tokens`` field counts tokens in *raw_output_for_tokens*
    when supplied (used by ``calibrate_normalize``, whose ``raw_output`` is a
    ``{"filter": …, "merge": …}`` dict).  For all other stages the field is
    derived from ``raw_output`` directly, using ``-1`` when it is falsy or
    non-string.

    Args:
        stage: Stage identifier string (``"extract"``, ``"anonymize"``, …).
        prompts: List of prompt dicts with ``role``/``path``/``sha``/``content``.
        raw_output: The verbatim model string(s) before post-processing.
            May be a dict for stages that run two LLM calls (normalize).
        parsed: Stage-specific parsed result dict.
        input_prompt_text: The prompt text to count for ``n_input_tokens``.
        measurement: Populated :class:`_Measurement` from
            :func:`_measured_local_call`.
        params: The :class:`CalibrateParams` from the request.
        state: The live server state dict (provides ``model_id``).
        supports_seed: ``True`` for local stages, ``False`` for SOTA stages
            (mirrors the existing ``_effective_params`` convention).
        raw_output_for_tokens: When provided, token counting uses this string
            instead of ``raw_output``.  Pass the primary model string for
            stages where ``raw_output`` is a dict (e.g. normalize's
            ``raw_filter``).  Defaults to ``raw_output`` when ``None``.
        **extra: Additional keys to merge into the response (e.g.
            ``phases=`` for the extract stage).

    Returns:
        Complete calibration response dict ready for JSON serialisation.
    """
    tokenizer = state.get("tokenizer")
    n_in = _count_tokens(tokenizer, input_prompt_text) if tokenizer else -1
    # n_output_tokens: prefer the caller-supplied string; fall back to raw_output
    # when it is a plain string.  Returns -1 for empty/non-string values.
    count_str: str
    if raw_output_for_tokens is not None:
        count_str = raw_output_for_tokens
    elif isinstance(raw_output, str):
        count_str = raw_output
    else:
        count_str = ""
    n_out = _count_tokens(tokenizer, count_str) if (tokenizer and count_str) else -1

    model_id = getattr(state.get("model_id"), "name", state.get("model_id", "unknown"))

    response: dict[str, Any] = {
        "stage": stage,
        "prompts": prompts,
        "raw_output": raw_output,
        "parsed": parsed,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "wall_clock_seconds": measurement.elapsed,
        "model": model_id,
        "params_effective": _effective_params(params, supports_seed=supports_seed),
        "vram_before": measurement.vram_before,
        "vram_after": measurement.vram_after,
    }
    response.update(extra)
    return response


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

    Routes through :meth:`ExtractionPipeline.run` (the single-topology
    chokepoint).  The pipeline lives on the process-wide
    ``ConsolidationLoop`` so calibration calls share the exact instance
    the production /consolidate cycle uses — same model, same config,
    same flags.  Never calls ``extract_graph`` directly.
    """
    _preflight(state)
    # Lazy-init the loop the same way the production /consolidate handler does.
    # This is not a parallel route — it's the same factory and the same single
    # ConsolidationLoop instance.  The FIRST call (calibrate or consolidate)
    # creates it; subsequent calls reuse.  Calibration touches only
    # ``loop.extraction`` (read-only): no merger, no trainer, no disk writes.
    loop = state.get("consolidation_loop")
    if loop is None:
        from paramem.server.consolidation import create_consolidation_loop

        loop = create_consolidation_loop(
            state["model"],
            state["tokenizer"],
            state["config"],
            state["memory_store"],
            state_provider=lambda: state,
        )
        state["consolidation_loop"] = loop
        state["model"] = loop.model

    # Resolve prompt files for transparency: even though the actual read
    # happens inside extract_graph via ``_load_prompt``, we surface the
    # operator-supplied path + sha + content in the response so dump
    # diffs show prompt provenance directly.  One prompt-pair is the
    # single ground truth for every source type — document chunks land
    # in the same ``{transcript}`` slot.
    user_filename = req.extraction_prompt_filename or "extraction.txt"
    sys_filename = req.extraction_system_prompt_filename or "extraction_system.txt"
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
    if req.params.seed is not None:
        overrides["seed"] = req.params.seed
    # NOTE: top_p, top_k do not flow through extract_graph's signature.
    # Document the limitation in params_effective.

    with _measured_local_call() as m:
        graph = loop.extraction.run(
            req.transcript,
            req.session_id,
            source_type=req.source_type,
            **overrides,
        )

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

    return _build_calibrate_response(
        stage="extract",
        prompts=[
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
        raw_output=raw_output,
        parsed=parsed,
        input_prompt_text=user_prompt_content,
        measurement=m,
        params=req.params,
        state=state,
        supports_seed=True,
        phases=phase_records,
    )


def calibrate_anonymize(state: dict, req: CalibrateAnonymizeRequest) -> dict[str, Any]:
    """Run the local anonymizer on an explicit graph + transcript."""
    _preflight(state)
    from paramem.graph.extractor import anonymize_with_local_model
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
    with _measured_local_call() as m:
        anon_facts, mapping, anon_transcript, raw_output = anonymize_with_local_model(
            graph,
            model,
            tokenizer,
            transcript=req.transcript,
            max_tokens=max_tokens,
            seed=req.params.seed,
        )

    parsed: dict[str, Any] = {
        "anonymized_facts": anon_facts,
        "mapping": mapping,
        "anonymized_transcript": anon_transcript,
    }

    return _build_calibrate_response(
        stage="anonymize",
        prompts=[
            {"role": "user", "path": prompt_path, "sha": prompt_sha, "content": prompt_content},
        ],
        raw_output=raw_output,
        parsed=parsed,
        input_prompt_text=prompt_content,
        measurement=m,
        params=req.params,
        state=state,
        supports_seed=True,
    )


def calibrate_plausibility(state: dict, req: CalibratePlausibilityRequest) -> dict[str, Any]:
    """Run the local plausibility filter on an explicit fact list."""
    _preflight(state)
    from paramem.graph.extractor import local_plausibility_filter

    model = state["model"]
    tokenizer = state["tokenizer"]

    filename = req.plausibility_prompt_filename or "sota_plausibility.txt"
    prompt_path, prompt_sha, prompt_content = _read_prompt(req.prompts_dir, filename)

    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    with _measured_local_call() as m:
        kept, raw_output = local_plausibility_filter(
            req.facts,
            req.transcript,
            model,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=req.params.seed,
        )

    dropped = [f for f in req.facts if f not in (kept or [])]
    parsed: dict[str, Any] = {
        "kept_facts": kept or [],
        "dropped_facts": dropped,
        "input_count": len(req.facts),
        "kept_count": len(kept or []),
        "dropped_count": len(dropped),
    }

    return _build_calibrate_response(
        stage="plausibility",
        prompts=[
            {"role": "user", "path": prompt_path, "sha": prompt_sha, "content": prompt_content},
        ],
        raw_output=raw_output,
        parsed=parsed,
        input_prompt_text=prompt_content,
        measurement=m,
        params=req.params,
        state=state,
        supports_seed=True,
    )


def calibrate_normalize(state: dict, req: CalibrateNormalizeRequest) -> dict[str, Any]:
    """Run the two-stage synonym-predicate dedup on an explicit relation list
    or a graph snapshot.

    Routes through :func:`paramem.graph.extractor.dedup_synonym_predicates_local`
    (the graph-layer primitive).  No merger/ledger state is modified — this is
    a read-only probe.

    The GPU lock and cuDNN deterministic flags are acquired here (not inside
    the primitive) so production callers can apply their own locking policy.
    """
    _preflight(state)

    # Validate mutually exclusive inputs.
    has_relations = req.relations is not None
    has_snapshot = req.snapshot_path is not None
    if has_relations == has_snapshot:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly one of 'relations' or 'snapshot_path' must be provided, "
                "not both and not neither."
            ),
        )

    from paramem.graph.extractor import dedup_synonym_predicates_local

    model = state["model"]
    tokenizer = state["tokenizer"]

    filter_filename = req.filter_prompt_filename or "graph_dedup_filter.txt"
    merge_filename = req.merge_prompt_filename or "graph_dedup_merge.txt"
    filter_path, filter_sha, filter_content = _read_prompt(req.prompts_dir, filter_filename)
    merge_path, merge_sha, merge_content = _read_prompt(req.prompts_dir, merge_filename)

    if has_relations:
        relations: list[dict] = req.relations  # type: ignore[assignment]
    else:
        # Load from a NetworkX node-link snapshot.
        import json as _json
        from pathlib import Path as _Path

        snap_path = _Path(req.snapshot_path)  # type: ignore[arg-type]
        if not snap_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"snapshot_path does not exist: {snap_path}",
            )
        try:
            snap = _json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read snapshot_path: {exc}",
            ) from exc
        # NetworkX node-link format: {"nodes": [...], "links": [...]} where
        # each link is {source, target, key, ...edge_data...}.
        links = snap.get("links", snap.get("edges", []))
        relations = []
        for link in links:
            if not isinstance(link, dict):
                continue
            pred = link.get("predicate")
            if not pred:
                continue
            relations.append(
                {
                    "subject": str(link.get("source", "")),
                    "predicate": str(pred),
                    "object": str(link.get("target", "")),
                }
            )

    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    from peft import PeftModel as _PeftModel

    with _measured_local_call() as m:
        if isinstance(model, _PeftModel):
            with model.disable_adapter():
                surviving, diag = dedup_synonym_predicates_local(
                    relations,
                    model,
                    tokenizer,
                    filter_prompt=filter_content,
                    merge_prompt=merge_content,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=req.params.seed,
                )
        else:
            surviving, diag = dedup_synonym_predicates_local(
                relations,
                model,
                tokenizer,
                filter_prompt=filter_content,
                merge_prompt=merge_content,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=req.params.seed,
            )

    raw_filter = diag.pop("raw_filter", "")
    raw_merge = diag.pop("raw_merge", "")
    parsed: dict[str, Any] = {
        "surviving_relations": surviving,
        "input_count": len(relations),
        "surviving_count": len(surviving),
        **diag,
    }

    return _build_calibrate_response(
        stage="normalize",
        prompts=[
            {
                "role": "filter",
                "path": filter_path,
                "sha": filter_sha,
                "content": filter_content,
            },
            {
                "role": "merge",
                "path": merge_path,
                "sha": merge_sha,
                "content": merge_content,
            },
        ],
        raw_output={"filter": raw_filter, "merge": raw_merge},
        parsed=parsed,
        input_prompt_text=filter_content,
        measurement=m,
        params=req.params,
        state=state,
        supports_seed=True,
        raw_output_for_tokens=raw_filter,
    )


def calibrate_name(state: dict, req: CalibrateNameRequest) -> dict[str, Any]:
    """Run the name-extraction LLM on an explicit turn list.

    Routes through :func:`paramem.graph.name_extraction.extract_name_via_llm`
    (the same function the production enrollment path uses).  The
    ``prompts_dir`` / filename fields let the calibration operator point at
    scratch copies of the prompt files; the module re-reads from disk on
    every call (no restart needed).

    ``user_turns_only`` mirrors the production default (``True``) — only
    user turns reach the model; set to ``False`` to include assistant turns
    and reproduce the original (buggy) context-scoping behaviour for
    comparative testing.

    Returns the uniform calibration shape: ``stage``, ``prompts``,
    ``raw_output`` (the verbatim model string BEFORE post-filters, so
    calibration sees what the model actually emitted), ``parsed``
    (``{"name": <str|None>}``), ``params_effective``, VRAM block, and
    wall-clock time.
    """
    _preflight(state)
    from paramem.graph.name_extraction import extract_name_via_llm

    model = state["model"]
    tokenizer = state["tokenizer"]

    # Resolve filenames with the same None-default + or-fallback convention
    # used by the other stages.
    sys_filename = req.name_system_prompt_filename or "name_extraction_system.txt"
    user_filename = req.name_prompt_filename or "name_extraction.txt"

    # Surface the prompt files for provenance — same approach as other stages.
    sys_path, sys_sha, sys_content = _read_prompt(req.prompts_dir, sys_filename)
    user_path, user_sha, user_content = _read_prompt(req.prompts_dir, user_filename)

    inference_params = {
        "temperature": req.params.temperature,
        "seed": req.params.seed,
        "max_tokens": req.params.max_tokens,
    }

    with _measured_local_call() as m:
        extracted, raw_output = extract_name_via_llm(
            req.turns,
            model,
            tokenizer,
            prompts_dir=req.prompts_dir,
            prompt_filename=user_filename,
            system_filename=sys_filename,
            user_turns_only=req.user_turns_only,
            params=inference_params,
        )

    parsed: dict[str, Any] = {"name": extracted}

    return _build_calibrate_response(
        stage="name",
        prompts=[
            {
                "role": "system",
                "path": sys_path,
                "sha": sys_sha,
                "content": sys_content,
            },
            {
                "role": "user",
                "path": user_path,
                "sha": user_sha,
                "content": user_content,
            },
        ],
        raw_output=raw_output,
        parsed=parsed,
        input_prompt_text=user_content,
        measurement=m,
        params=req.params,
        state=state,
        supports_seed=True,
    )


def _effective_params(params: CalibrateParams, *, supports_seed: bool) -> dict:
    """Return the params dict the call effectively applied.

    ``supports_seed`` distinguishes local stages (where seed is honoured
    via a scoped torch.Generator) from SOTA stages (where Anthropic's API
    accepts no seed parameter and the field is silently dropped).  The
    response uses this to inform the operator which fields actually
    landed.

    seed threads through all three local stages (extract, anonymize,
    plausibility) via the helpers' ``seed`` parameter forwarded to
    ``generate_answer``.  top_p / top_k are not yet threaded for these
    stages (documented gap; SOTA stages follow the Anthropic API which
    omits them).  The field is reported as-requested for transparency.
    """
    out: dict = {}
    for f in ("temperature", "top_p", "top_k", "max_tokens"):
        out[f] = getattr(params, f)
    out["seed"] = params.seed if supports_seed else None
    return out
