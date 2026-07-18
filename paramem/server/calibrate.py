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
  the caller-supplied SessionGraph + transcript, returning the model's
  ``real_name -> placeholder`` mapping AND its own ``anonymized_transcript``
  rewrite (the model is the sole scope authority) — no facts,
  the anonymizer never authors those.  ``scrub`` is sourced from the
  server's configured ``SanitizationConfig.scrub``, not a request field
  (see :func:`calibrate_anonymize`).
* ``POST /calibrate/plausibility`` — runs ``local_plausibility_filter``
  on the caller-supplied fact list + transcript.
* ``POST /calibrate/enrich`` — runs ``_filter_with_sota`` (the production
  ``sota_enrich`` stage) on the caller-supplied anonymized fact list +
  anonymized transcript.  Makes a BILLED call to the configured SOTA
  provider (see :func:`calibrate_enrich`).

Returns a uniform shape:

  prompt_path, prompt_sha, prompt_template, raw_output, parsed,
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

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from paramem.graph.phase_trace import PhaseRecord, extraction_trace, phase_trace
from paramem.graph.prompts import _DEFAULT_PROMPT_DIR
from paramem.server.gpu_lock import gpu_lock_sync
from paramem.server.session_buffer import SessionBuffer

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


class CalibrateProceduralRequest(BaseModel):
    """Run the procedural (preferences/habits) extractor on a transcript.

    Mirrors :class:`CalibrateExtractRequest`'s prompt-source + seed
    configuration vocabulary — ``prompts_dir`` / ``procedural_prompt_filename``
    / ``procedural_system_prompt_filename`` override the prompt pair,
    ``params.seed`` threads through :meth:`ExtractionPipeline.run_procedural`.
    ``stop_phase`` is intentionally NOT mirrored: procedural extraction is a
    single-phase call, so a stop point is inapplicable.
    """

    transcript: str
    speaker_id: str
    speaker_name: str | None = None
    source_type: str = Field(default="transcript", pattern="^(transcript|document)$")
    session_id: str = "calib"
    prompts_dir: str | None = None
    procedural_prompt_filename: str | None = None
    procedural_system_prompt_filename: str | None = None
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


class CalibrateEnrichRequest(BaseModel):
    """Run the production SOTA enrichment stage on an explicit fact list + transcript.

    ``facts`` are the ANONYMIZED facts (the calibration client typically
    takes these from a prior ``/calibrate/anonymize`` response, built via
    the production ``_build_anon_facts`` primitive) — this endpoint does
    not de-anonymize anything itself. ``transcript`` is the anonymized
    transcript, turn-marked the same way every other ``/calibrate/*``
    transcript field is (see :func:`_require_turn_marked_transcript`).

    Runs the PRODUCTION ``sota_enrich`` stage
    (:func:`paramem.graph.extractor._filter_with_sota`) — the same
    coreference-resolution + compound-splitting + reification call
    :func:`~paramem.graph.extractor._sota_pipeline` makes every
    consolidation cycle. This is a BILLED cloud call to the configured
    SOTA provider.

    This is the PER-SESSION enrichment stage only (facts + transcript
    input). It is NOT the graph-level SOTA enrichment
    (:func:`~paramem.graph.extractor._graph_enrich_with_sota`, prompt
    ``sota_graph_enrichment.txt``), which runs on the accumulated graph
    with no transcript and has no calibrate endpoint of its own.
    """

    facts: list[dict]
    transcript: str
    prompts_dir: str | None = None
    enrichment_prompt_filename: str | None = None
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateNormalizeRequest(BaseModel):
    """Run predicate normalization on an explicit relation list
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

    ``normalization_prompt_filename`` defaults to ``predicate_normalization.txt``.
    ``prompts_dir`` defaults to the project's ``configs/prompts/`` directory.
    """

    relations: list[dict] | None = None
    snapshot_path: str | None = None
    normalization_prompt_filename: str | None = None
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


def _ensure_prompt_exists(prompts_dir: str | Path | None, filename: str) -> None:
    """Guard-time existence check for an operator-overridable prompt file.

    Runs from a handler's ``guard`` closure — BEFORE any model call — so a
    typo'd ``prompts_dir``/filename override surfaces as HTTP 400 with zero
    inference cost.

    The no-``prompts_dir`` default anchors to
    :data:`paramem.graph.prompts._DEFAULT_PROMPT_DIR` — the same
    project-root-anchored directory :func:`~paramem.graph.prompts._load_prompt`
    searches — so a no-override calibration call checks the real shipped
    default regardless of the process's current working directory.

    Beyond that anchor, resolution is STRICT (owner-approved semantics):
    an operator-supplied ``prompts_dir`` is checked in isolation, with NO
    silent fall-through to the project default — unlike ``_load_prompt``
    itself, which searches ``[prompts_dir, _DEFAULT_PROMPT_DIR]`` and would
    let a missing override silently resolve to the production prompt of
    the same name. Provenance (path/sha/template) for a file that DOES
    exist is no longer built here; it comes from the phase-trace records
    :func:`_run_calibration` reads after ``dispatch`` runs (``_load_prompt``'s
    own :func:`~paramem.graph.phase_trace.record_prompt` call).
    """
    base = Path(prompts_dir) if prompts_dir is not None else _DEFAULT_PROMPT_DIR
    path = base / filename
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Prompt file not found: {path}. "
            f"Calibration requires the operator to specify a real prompt "
            f"file; embedded fallbacks are not used.",
        )


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
# Shared measurement primitives
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
# Turn-marking gate — every calibrate endpoint that accepts a ``transcript``
# ---------------------------------------------------------------------------


def _production_turn_markers() -> tuple[str, ...]:
    """The exact ``[<role>]`` marker prefixes production transcripts begin
    with, derived by calling the SAME renderer every producer uses —
    :meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`
    (``/chat`` user + assistant turns, document ingest, and cloud-egress
    anonymization all render through it; see that method's docstring).

    ``"user"`` and ``"assistant"`` are the only two roles any production
    caller ever passes to ``SessionBuffer.append`` /
    ``append_document_chunk`` (both documented on those methods).  Calling
    the real renderer for each — rather than hardcoding ``"[user]"`` /
    ``"[assistant]"`` here — means this module carries no second copy of
    the marker shape; if ``_format_turns`` ever changes its bracket
    convention, this list changes with it automatically.
    """
    markers = []
    for role in ("user", "assistant"):
        lines, _ = SessionBuffer._format_turns([{"role": role, "text": "x"}])
        marker, _sep, _rest = lines[0].partition(" ")
        markers.append(marker)
    return tuple(markers)


def _require_turn_marked_transcript(transcript: str) -> None:
    """Fail loud (HTTP 400) when ``transcript`` is not the production
    turn-marked surface.

    Every extraction/anonymization/plausibility prompt's few-shots
    (``configs/prompts/extraction.txt``, ``anonymization.txt``, …) are
    calibrated exclusively on the ``[user] <text>`` / ``[assistant]
    <text>`` surface :meth:`SessionBuffer._format_turns` renders in
    production (``/chat``, document ingest, cloud egress). A bare,
    unmarked transcript puts the model off-distribution from every
    example it was tuned on — this is exactly how the ``Pat's dog``
    cloud-egress leak stayed invisible: the calibration endpoint that
    exists to tune these prompts was itself feeding them a surface
    production never sends.

    This is a CHECK, not a repair: an unmarked transcript is an operator
    error, so it is rejected with a message naming the expected surface —
    never silently prepended, never guessed.
    """
    markers = _production_turn_markers()
    if not transcript.startswith(markers):
        raise HTTPException(
            status_code=400,
            detail=(
                f"transcript must be turn-marked — it must start with one "
                f"of {markers!r} (the same surface "
                f"SessionBuffer._format_turns renders for production "
                f"/chat, document ingest, and cloud egress; see "
                f"DEPLOYMENT.md 'Calibration loop'). Got: {transcript[:80]!r}"
            ),
        )


# ---------------------------------------------------------------------------
# The shared calibration primitive
# ---------------------------------------------------------------------------


def _provenance_from_records(
    records: list[PhaseRecord], phase: str
) -> tuple[list[dict[str, Any]], str]:
    """Derive a calibration response's prompt provenance from phase-trace records.

    ``prompts``: every ``{path, sha, template}`` entry any record in
    ``records`` captured, in firing order, DEDUPED by ``(path, sha)``
    (first occurrence wins). A multi-phase run (e.g. ``local_extract`` +
    ``second_order_extract``, both re-loading ``extraction_system.txt``)
    would otherwise list the same system prompt twice.

    ``input_prompt_text``: the ``template`` of the entry belonging to the
    record named ``phase`` that is NOT a system prompt — selected
    null-safely via ``not (p.get("path") or "").endswith("_system.txt")``.
    Falls back to ``""`` when ``phase`` has no matching record, or that
    record captured no non-system prompt (e.g. procedural extraction on
    an empty transcript, or an opted-out anonymize call that never
    reaches a model). A well-formed, empty-provenance response is
    preferable to a crash on these legitimate no-model-call paths.

    Args:
        records: Typed phase records from ``ExtractionTrace.records``
            (one :class:`ExtractionTrace` per :func:`_run_calibration` call).
        phase: The :data:`~paramem.graph.phase_trace.PHASE_NAMES` phase
            whose user-prompt template feeds ``n_input_tokens``.

    Returns:
        ``(prompts, input_prompt_text)``.
    """
    prompts: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for record in records:
        for p in record.prompts or []:
            key = (p.get("path"), p.get("sha"))
            if key in seen:
                continue
            seen.add(key)
            prompts.append(p)

    input_prompt_text = ""
    for record in records:
        if record.name != phase:
            continue
        entry = next(
            (
                p
                for p in (record.prompts or [])
                if not (p.get("path") or "").endswith("_system.txt")
            ),
            None,
        )
        if entry is not None:
            input_prompt_text = entry.get("template") or ""
        break
    return prompts, input_prompt_text


def _run_calibration(
    *,
    stage: str,
    entry: str,
    phase: str | None,
    guard,
    dispatch,
    input_prompt_phase: str,
    state: dict,
    params: CalibrateParams,
    supports_seed: bool,
) -> dict[str, Any]:
    """Shared execution + response-assembly primitive for every ``/calibrate/*`` handler.

    Each handler shrinks to building a ``guard`` closure (raises HTTP 400
    for any invalid input, BEFORE any model call) and a ``dispatch``
    closure (runs the step, returns ``(raw_output, parsed)``), then calls
    this function. Prompt provenance is read uniformly from the phase
    trace — see :func:`_provenance_from_records` — never hand-built.

    Execution order:

    1. :func:`_preflight` — the shared 404/503 gates.
    2. ``guard()`` — MUST run before any model call / before the trace
       even opens, so a 400 costs zero inference.
    3. ``with extraction_trace() as trace:`` — opens (or, if already
       active, no-ops onto) the trace every :func:`~paramem.graph.prompts._load_prompt`
       call records onto.
    4. ``with _measured_local_call() as m:`` — GPU lock + timing/VRAM,
       shared by every local stage.
    5. ``entry == "standalone"``: wrap ``dispatch()`` in
       ``with phase_trace(phase):`` so the step's own prompt loads land on
       a named phase record. ``entry == "pipeline"``: call ``dispatch()``
       bare — the pipeline (``extract_graph`` et al.) opens its own named
       phases onto the same outer trace; nothing here would open a second,
       redundant phase.
    6. Assembles the uniform 11-key response (+ ``phases``) directly from
       ``records``, ``prompts``, ``input_prompt_text``, ``raw_output``,
       ``parsed``, and ``m`` — the per-stage parts (``prompts``,
       ``raw_output``, ``parsed``) come from steps 1-5 above; the shared
       envelope (token counts, wall clock, VRAM, ``model``,
       ``params_effective``) is assembled here so every stage reports it
       identically.  ``n_output_tokens`` is derived from ``raw_output``
       directly, using ``-1`` when it is falsy or non-string — no
       calibration stage needs a token count over a different string.

    Args:
        stage: Response ``"stage"`` label (``"plausibility"``, ``"normalize"``, …).
        entry: ``"standalone"`` or ``"pipeline"`` — selects step 5 above.
        phase: The :data:`~paramem.graph.phase_trace.PHASE_NAMES` phase to
            open for a standalone step. ``None`` for a pipeline entry
            (the pipeline names its own phases).
        guard: Zero-arg callable raising :class:`~fastapi.HTTPException`
            (400) for any invalid input. Called before the trace opens.
        dispatch: Zero-arg callable that runs the step and returns
            ``(raw_output, parsed)``.
        input_prompt_phase: Which phase record's non-system prompt
            template feeds ``n_input_tokens`` (see
            :func:`_provenance_from_records`).
        state: The live server state dict.
        params: The request's :class:`CalibrateParams`.
        supports_seed: ``True`` for local stages, ``False`` for SOTA
            stages (mirrors the existing ``_effective_params`` convention).

    Returns:
        The uniform calibration response dict.
    """
    _preflight(state)
    guard()
    with extraction_trace() as trace:
        with _measured_local_call() as m:
            if entry == "standalone":
                with phase_trace(phase):
                    raw_output, parsed = dispatch()
            elif entry == "pipeline":
                # The pipeline (extract_graph et al.) opens its own named
                # phases onto this same outer trace via the
                # extraction_trace() nesting no-op — nothing to wrap here.
                raw_output, parsed = dispatch()
            else:
                raise ValueError(f"Unknown entry {entry!r}; expected 'standalone' or 'pipeline'.")
    records = trace.records
    prompts, input_prompt_text = _provenance_from_records(records, input_prompt_phase)

    tokenizer = state.get("tokenizer")
    n_in = _count_tokens(tokenizer, input_prompt_text) if tokenizer else -1
    count_str = raw_output if isinstance(raw_output, str) else ""
    n_out = _count_tokens(tokenizer, count_str) if (tokenizer and count_str) else -1
    model_id = getattr(state.get("model_id"), "name", state.get("model_id", "unknown"))

    return {
        "stage": stage,
        "prompts": prompts,
        "raw_output": raw_output,
        "parsed": parsed,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "wall_clock_seconds": m.elapsed,
        "model": model_id,
        "params_effective": _effective_params(params, supports_seed=supports_seed),
        "vram_before": m.vram_before,
        "vram_after": m.vram_after,
        "phases": [r.to_dict() for r in records],
    }


# ---------------------------------------------------------------------------
# Stage handlers — invoked from the registered FastAPI routes in app.py
# ---------------------------------------------------------------------------


def _ensure_calibration_loop(state: dict):
    """Lazy-init the loop the same way the production /consolidate handler does.

    This is not a parallel route — it's the same factory and the same single
    ``ConsolidationLoop`` instance.  The FIRST call (calibrate or consolidate)
    creates it; subsequent calls reuse.  Calibration touches only
    ``loop.extraction`` (read-only): no merger, no trainer, no disk writes.
    Shared by every calibration handler that needs the pipeline — do not
    inline a second init path.
    """
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
    return loop


def calibrate_extract(state: dict, req: CalibrateExtractRequest) -> dict[str, Any]:
    """Run the local extractor for a single transcript.

    Thin wrapper around :func:`_run_calibration` with ``entry="pipeline"``:
    ``guard`` performs the turn-marking + prompt-file checks (zero
    inference cost on failure); ``dispatch`` routes through
    :meth:`ExtractionPipeline.run` (the single-topology chokepoint) on the
    process-wide ``ConsolidationLoop`` so calibration calls share the
    exact instance the production /consolidate cycle uses — same model,
    same config, same flags.  Never calls ``extract_graph`` directly.
    Prompt provenance and ``n_input_tokens`` come from the
    ``local_extract`` phase record the pipeline opens on its own trace
    (see :func:`_provenance_from_records`) — never a hand-built
    ``prompts=[...]`` literal.
    """
    user_filename = req.extraction_prompt_filename or "extraction.txt"
    sys_filename = req.extraction_system_prompt_filename or "extraction_system.txt"
    # Closure holder so the debug-writer call after _run_calibration
    # returns can reach the loop dispatch() resolved.
    holder: dict[str, Any] = {}

    def guard() -> None:
        _require_turn_marked_transcript(req.transcript)
        _ensure_prompt_exists(req.prompts_dir, user_filename)
        _ensure_prompt_exists(req.prompts_dir, sys_filename)

    def dispatch() -> tuple[Any, dict]:
        loop = _ensure_calibration_loop(state)
        holder["loop"] = loop

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

        graph = loop.extraction.run(
            req.transcript,
            req.session_id,
            source_type=req.source_type,
            **overrides,
        )
        parsed = graph.model_dump(mode="json") if hasattr(graph, "model_dump") else {}

        # Local-extract raw output, surfaced at the top level for easy
        # diffing of the local-extract prompt without traversing the
        # phases list.
        from paramem.graph.phase_trace import get_phases

        phase_records = [r.to_dict() for r in get_phases(graph)]
        local_extract = next(
            (p for p in phase_records if p.get("name") == "local_extract"),
            None,
        )
        raw_output = (local_extract or {}).get("raw_output") or ""
        return raw_output, parsed

    response = _run_calibration(
        stage="extract",
        entry="pipeline",
        phase=None,
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="local_extract",
        state=state,
        params=req.params,
        supports_seed=True,
    )
    # Debug artifact: the full result (parsed graph incl. diagnostics — e.g.
    # entity_correction_verdicts, which carries apply-gate-REJECTED entity
    # correction proposals that never reach graph.diagnostics["entity_corrections"]
    # — plus phase records and raw output) so rejected proposals stay
    # inspectable. Self-gated; no-op unless config.debug is on.
    holder["loop"]._debug_writer.on_calibrate_extract(response, session_id=req.session_id)
    return response


def calibrate_procedural(state: dict, req: CalibrateProceduralRequest) -> dict[str, Any]:
    """Run the procedural extractor for a single transcript.

    Thin wrapper around :func:`_run_calibration` with ``entry="pipeline"``:
    ``guard`` performs the speaker-id, turn-marking, and prompt-file
    checks (zero inference cost on failure); ``dispatch`` routes through
    :meth:`ExtractionPipeline.run_procedural` (the single-topology
    chokepoint for procedural extraction) on the same process-wide
    ``ConsolidationLoop`` that :func:`calibrate_extract` shares — same
    model, same config, same flags.  Never calls ``extract_procedural_graph``
    directly.

    ``extract_procedural_graph`` self-traces the ``procedural_extract``
    phase via the shared ``_run_local_extraction`` primitive (the same one
    ``local_extract``/``second_order_extract`` use), so ``raw_output`` and
    prompt provenance are surfaced from that phase record exactly like the
    other pipeline stages (see :func:`_provenance_from_records`) — never a
    hand-built ``prompts=[...]`` literal.
    """
    user_filename = req.procedural_prompt_filename or "extraction_procedural.txt"
    sys_filename = req.procedural_system_prompt_filename or "extraction_system.txt"

    def guard() -> None:
        if not req.speaker_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "speaker_id is required for procedural extraction (no empty-string default)."
                ),
            )
        _require_turn_marked_transcript(req.transcript)
        _ensure_prompt_exists(req.prompts_dir, user_filename)
        _ensure_prompt_exists(req.prompts_dir, sys_filename)

    def dispatch() -> tuple[Any, dict]:
        loop = _ensure_calibration_loop(state)
        graph = loop.extraction.run_procedural(
            req.transcript,
            req.session_id,
            speaker_id=req.speaker_id,
            speaker_name=req.speaker_name,
            source_type=req.source_type,
            prompts_dir=req.prompts_dir,
            system_prompt_filename=sys_filename,
            user_prompt_filename=user_filename,
            seed=req.params.seed,
        )
        parsed = graph.model_dump(mode="json") if hasattr(graph, "model_dump") else {}

        from paramem.graph.phase_trace import get_phases

        phase_records = [r.to_dict() for r in get_phases(graph)]
        procedural_extract = next(
            (p for p in phase_records if p.get("name") == "procedural_extract"),
            None,
        )
        raw_output = (procedural_extract or {}).get("raw_output") or ""
        return raw_output, parsed

    return _run_calibration(
        stage="procedural",
        entry="pipeline",
        phase=None,
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="procedural_extract",
        state=state,
        params=req.params,
        supports_seed=True,
    )


def calibrate_anonymize(state: dict, req: CalibrateAnonymizeRequest) -> dict[str, Any]:
    """Run the local anonymizer on an explicit graph + transcript.

    Thin wrapper around :func:`_run_calibration`: ``guard`` performs the
    turn-marking + prompt-file checks (zero inference cost on failure),
    then validates the ``graph`` payload into a :class:`SessionGraph` and
    stashes it in a closure-local dict so ``dispatch`` (called after
    ``guard``) can read it back.  Prompt provenance and ``n_input_tokens``
    come from the ``anonymize`` phase record :func:`_run_calibration`
    opens around ``dispatch`` (the ``load_anonymization_prompt`` /
    ``_load_prompt`` calls inside ``anonymize_with_local_model`` record
    onto it) — never a hand-built ``prompts=[...]`` literal.

    ``scrub`` — the operator's PII-vocabulary hint list, the sole scope
    authority the model classifies against — is sourced from the
    server's live ``SanitizationConfig.scrub``
    (``state["config"].sanitization.scrub``), the SAME policy knob every
    production call site reads (``ExtractionPipeline.kwargs``,
    ``extract_and_anonymize_for_cloud``, ``_sota_pipeline``).  It is
    deliberately NOT a request field: every other ``/calibrate/*``
    request injects operator PROMPTS/PARAMS (filenames, seed,
    temperature, ...), never privacy POLICY —
    :class:`CalibrateExtractRequest` exposes no ``sota_enabled``- or
    ``scrub``-equivalent override either.  A per-request override here
    would let calibration silently anonymize against a scope the
    operator never configured on a security-critical egress path.  When
    ``scrub`` is empty, ``dispatch`` makes no model call, so the
    ``anonymize`` phase records no prompts — a well-formed, empty
    provenance response is correct here, not synthesized.

    Reports the model's two artifacts under the current
    ``anonymize_with_local_model`` contract — ``mapping`` and
    ``anonymized_transcript`` — via ``parsed``.  ``parsed["status"]``
    reuses the same three-state vocabulary production diagnostics record
    (``paramem.graph.extractor``'s ``graph.diagnostics["anonymize"]``):

    * ``"opted_out"`` — ``scrub`` is empty (operator opt-out): no
      model call is made; ``anonymized_transcript`` is the passed-in
      ``req.transcript`` verbatim and ``mapping`` is ``{}``.
    * ``"failed"`` — fail-closed: the model call returned a parse
      failure OR a missing/empty ``anonymized_transcript``.  ``mapping``
      is reported as ``{}`` and ``anonymized_transcript`` as ``""`` —
      this endpoint never falls back to the real-name transcript to
      manufacture an apparent success.
    * ``"ok"`` — the model returned both artifacts.
    """
    from paramem.graph.extractor import anonymize_with_local_model
    from paramem.graph.schema import SessionGraph
    from paramem.models.loader import base_model_inference

    filename = req.anonymization_prompt_filename or "anonymization.txt"
    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192

    resolved: dict[str, Any] = {}

    def guard() -> None:
        _require_turn_marked_transcript(req.transcript)
        _ensure_prompt_exists(req.prompts_dir, filename)
        try:
            resolved["graph"] = SessionGraph.model_validate(req.graph)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SessionGraph payload: {exc}",
            ) from exc

    def dispatch() -> tuple[Any, dict]:
        model = state["model"]
        tokenizer = state["tokenizer"]
        scrub = set(state["config"].sanitization.scrub)
        graph = resolved["graph"]

        if not scrub:
            # Operator opt-out: no anonymizer call; the transcript
            # egresses verbatim, sourced from the passed-in transcript —
            # never a model artifact.
            status, mapping, anonymized_transcript, raw_output = (
                "opted_out",
                {},
                req.transcript,
                "",
            )
        else:
            with base_model_inference(model):
                mapping, anonymized_transcript, raw_output = anonymize_with_local_model(
                    graph,
                    model,
                    tokenizer,
                    transcript=req.transcript,
                    scrub=scrub,
                    max_tokens=max_tokens,
                    seed=req.params.seed,
                    prompts_dir=req.prompts_dir,
                    prompt_filename=filename,
                )
            if mapping is None:
                # Fail-closed: parse failure OR missing/empty
                # anonymized_transcript.  Never fall back to the
                # real-name transcript.
                status, mapping, anonymized_transcript = "failed", {}, ""
            else:
                status = "ok"

        parsed: dict[str, Any] = {
            "status": status,
            "mapping": mapping,
            "anonymized_transcript": anonymized_transcript,
        }
        return raw_output, parsed

    return _run_calibration(
        stage="anonymize",
        entry="standalone",
        phase="anonymize",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="anonymize",
        state=state,
        params=req.params,
        supports_seed=True,
    )


def calibrate_plausibility(state: dict, req: CalibratePlausibilityRequest) -> dict[str, Any]:
    """Run the local plausibility filter on an explicit fact list.

    Thin wrapper around :func:`_run_calibration`: ``guard`` performs the
    turn-marking + prompt-file checks (zero inference cost on failure);
    ``dispatch`` runs :func:`~paramem.graph.extractor.local_plausibility_filter`
    under ``base_model_inference`` (base weights, matching production).
    Prompt provenance and ``n_input_tokens`` come from the
    ``deanon_plausibility`` phase record :func:`_run_calibration` opens
    around ``dispatch`` — never a hand-built ``prompts=[...]`` literal.
    """
    from paramem.graph.extractor import local_plausibility_filter
    from paramem.models.loader import base_model_inference

    model = state.get("model")
    tokenizer = state.get("tokenizer")
    filename = req.plausibility_prompt_filename or "sota_plausibility.txt"
    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    def guard() -> None:
        _require_turn_marked_transcript(req.transcript)
        _ensure_prompt_exists(req.prompts_dir, filename)

    def dispatch() -> tuple[Any, dict]:
        with base_model_inference(model):
            kept, raw_output = local_plausibility_filter(
                req.facts,
                req.transcript,
                model,
                tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=req.params.seed,
                prompts_dir=req.prompts_dir,
                prompt_filename=filename,
            )
        dropped = [f for f in req.facts if f not in (kept or [])]
        parsed: dict[str, Any] = {
            "kept_facts": kept or [],
            "dropped_facts": dropped,
            "input_count": len(req.facts),
            "kept_count": len(kept or []),
            "dropped_count": len(dropped),
        }
        return raw_output, parsed

    return _run_calibration(
        stage="plausibility",
        entry="standalone",
        phase="deanon_plausibility",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="deanon_plausibility",
        state=state,
        params=req.params,
        supports_seed=True,
    )


def calibrate_enrich(state: dict, req: CalibrateEnrichRequest) -> dict[str, Any]:
    """Run the production SOTA enrichment stage on an explicit fact list + transcript.

    This calibrates the PER-SESSION ``sota_enrich`` stage
    (:func:`~paramem.graph.extractor._filter_with_sota`) only — it does
    NOT reach the separate graph-level SOTA enrichment
    (:func:`~paramem.graph.extractor._graph_enrich_with_sota`, called
    from ``_run_graph_enrichment`` in consolidation.py, prompt
    ``sota_graph_enrichment.txt``), which has no calibrate endpoint.

    Thin wrapper around :func:`_run_calibration`: ``guard`` performs the
    turn-marking + prompt-file checks (zero inference cost on failure),
    then resolves the SOTA provider config and API key — a missing
    provider config or unset key surfaces as HTTP 400 before any cloud
    call is made. ``dispatch`` runs
    :func:`~paramem.graph.extractor._filter_with_sota` (the SAME function
    :func:`~paramem.graph.extractor._sota_pipeline` calls every
    consolidation cycle) directly — no mirrored cloud-call
    reimplementation. Prompt provenance and ``n_input_tokens`` come from
    the ``sota_enrich`` phase record :func:`_run_calibration` opens around
    ``dispatch`` (``_filter_with_sota`` loads its user prompt at call time
    via ``_load_prompt``, which records onto that phase) — never a
    hand-built ``prompts=[...]`` literal.

    Makes a BILLED cloud call to the configured SOTA provider
    (``consolidation.extraction_noise_filter`` in ``server.yaml``).
    ``supports_seed=False`` — matching the SOTA-stage convention in
    :func:`_effective_params`, since the provider APIs this call reaches
    accept no seed parameter.

    ``_measured_local_call`` still wraps this call (GPU lock + VRAM/timing
    snapshot) for response-shape consistency with every other calibration
    stage, even though the call itself is cloud-side and touches no VRAM.
    """
    from paramem.graph.extractor import (
        _DEFAULT_FILTER_MAX_TOKENS,
        _filter_with_sota,
        _resolve_sota_api_key,
    )

    filename = req.enrichment_prompt_filename or "sota_enrichment.txt"
    max_tokens = (
        req.params.max_tokens if req.params.max_tokens is not None else _DEFAULT_FILTER_MAX_TOKENS
    )
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    resolved: dict[str, Any] = {}

    def guard() -> None:
        _require_turn_marked_transcript(req.transcript)
        _ensure_prompt_exists(req.prompts_dir, filename)
        cfg = state["config"]
        provider = cfg.consolidation.extraction_noise_filter
        if not provider:
            raise HTTPException(
                status_code=400,
                detail="SOTA enrichment provider not configured — set "
                "consolidation.extraction_noise_filter in server.yaml.",
            )
        api_key = _resolve_sota_api_key(provider)
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=f"No API key for SOTA provider {provider!r} "
                f"(set the provider's key env var).",
            )
        resolved["provider"] = provider
        resolved["filter_model"] = cfg.consolidation.extraction_noise_filter_model
        resolved["endpoint"] = cfg.consolidation.extraction_noise_filter_endpoint or None
        resolved["api_key"] = api_key

    def dispatch() -> tuple[Any, dict]:
        facts, updated_transcript, bindings, raw, info = _filter_with_sota(
            req.facts,
            resolved["api_key"],
            resolved["provider"],
            resolved["filter_model"],
            req.transcript,
            endpoint=resolved["endpoint"],
            max_tokens=max_tokens,
            temperature=temperature,
            prompts_dir=req.prompts_dir,
            prompt_filename=filename,
        )
        parsed: dict[str, Any] = {
            "facts": facts if facts is not None else [],
            "updated_transcript": updated_transcript,
            "bindings": bindings,
            "info": info,
        }
        return raw or "", parsed

    return _run_calibration(
        stage="enrich",
        entry="standalone",
        phase="sota_enrich",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="sota_enrich",
        state=state,
        params=req.params,
        supports_seed=False,
    )


def calibrate_normalize(state: dict, req: CalibrateNormalizeRequest) -> dict[str, Any]:
    """Run single-stage synonym-predicate clustering on an explicit relation list
    or a graph snapshot.

    Thin wrapper around :func:`_run_calibration`. ``guard`` performs, in
    order: the mutually-exclusive ``relations``/``snapshot_path`` 400, the
    prompt-file check, and — for the ``snapshot_path`` branch — the
    NetworkX node-link load + flatten (with its own missing/unreadable-file
    400s). The resolved relation list is stashed in a closure-local dict so
    ``dispatch`` (called after ``guard``) can read it back.

    Routes through :func:`paramem.graph.extractor.normalize_predicates`
    (the graph-layer primitive).  No merger/ledger state is modified — this
    is a read-only probe.  A code survivor (first grounded cluster member)
    is applied locally to produce ``surviving_relations`` for inspection.
    """
    from paramem.graph.extractor import normalize_predicates
    from paramem.graph.name_match import canonical as _can
    from paramem.models.loader import base_model_inference

    model = state.get("model")
    tokenizer = state.get("tokenizer")
    prompt_filename = req.normalization_prompt_filename or "predicate_normalization.txt"
    max_tokens = req.params.max_tokens if req.params.max_tokens is not None else 8192
    temperature = req.params.temperature if req.params.temperature is not None else 0.0

    resolved: dict[str, list[dict]] = {}

    def guard() -> None:
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
        _ensure_prompt_exists(req.prompts_dir, prompt_filename)

        if has_relations:
            resolved["relations"] = req.relations  # type: ignore[assignment]
            return

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
        relations: list[dict] = []
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
        resolved["relations"] = relations

    def dispatch() -> tuple[Any, dict]:
        relations = resolved["relations"]
        with base_model_inference(model):
            clusters_by_so, diag = normalize_predicates(
                relations,
                model=model,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=req.params.seed,
                prompts_dir=req.prompts_dir,
                prompt_filename=prompt_filename,
            )

        # Apply code survivor (first grounded cluster member) to produce
        # surviving_relations for read-only inspection.
        # NOTE: production uses MAX reinforcement_count as survivor, not
        # first-in-cluster; surviving_relations illustrates which predicates
        # cluster, not which would survive in production.
        retire_set: dict[tuple[str, str], set[str]] = {}
        for (can_s, can_o), group_clusters in clusters_by_so.items():
            for cluster in group_clusters:
                if len(cluster) >= 2:
                    for retired_pred in cluster[1:]:
                        retire_set.setdefault((can_s, can_o), set()).add(retired_pred)

        surviving = [
            rel
            for rel in relations
            if _can(str(rel.get("predicate", "")))
            not in retire_set.get(
                (_can(str(rel.get("subject", ""))), _can(str(rel.get("object", "")))), set()
            )
        ]

        raw_outputs = diag.pop("raw_outputs", [])
        raw_output_str = "\n---\n".join(raw_outputs) if raw_outputs else ""
        parsed: dict[str, Any] = {
            "surviving_relations": surviving,
            "input_count": len(relations),
            "surviving_count": len(surviving),
            **diag,
        }
        return raw_output_str, parsed

    return _run_calibration(
        stage="normalize",
        entry="standalone",
        phase="normalize",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="normalize",
        state=state,
        params=req.params,
        supports_seed=True,
    )


def calibrate_name(state: dict, req: CalibrateNameRequest) -> dict[str, Any]:
    """Run the name-extraction LLM on an explicit turn list.

    Thin wrapper around :func:`_run_calibration`. ``guard`` checks both
    prompt files (system + user) exist before any model call; ``dispatch``
    runs :func:`~paramem.graph.name_extraction.extract_name_via_llm` (the
    same function the production enrollment path uses) under
    ``base_model_inference``. Prompt provenance and ``n_input_tokens`` come
    from the ``name_extract`` phase record :func:`_run_calibration` opens
    around ``dispatch`` — never a hand-built ``prompts=[...]`` literal.

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
    from paramem.graph.name_extraction import extract_name_via_llm
    from paramem.models.loader import base_model_inference

    model = state.get("model")
    tokenizer = state.get("tokenizer")

    # Resolve filenames with the same None-default + or-fallback convention
    # used by the other stages.
    sys_filename = req.name_system_prompt_filename or "name_extraction_system.txt"
    user_filename = req.name_prompt_filename or "name_extraction.txt"

    inference_params = {
        "temperature": req.params.temperature,
        "seed": req.params.seed,
        "max_tokens": req.params.max_tokens,
    }

    def guard() -> None:
        _ensure_prompt_exists(req.prompts_dir, sys_filename)
        _ensure_prompt_exists(req.prompts_dir, user_filename)

    def dispatch() -> tuple[Any, dict]:
        with base_model_inference(model):
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
        return raw_output, parsed

    return _run_calibration(
        stage="name",
        entry="standalone",
        phase="name_extract",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="name_extract",
        state=state,
        params=req.params,
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
