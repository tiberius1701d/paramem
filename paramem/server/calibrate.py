"""Calibration endpoints for live prompt iteration.

Exposes probes against the live Mistral instance the production cycle
uses — same model, same VRAM/allocator history, same prompt-loading
mechanism.  Results are a 1:1 reflection of what the production pipeline
would emit on this input given this prompt.

**1:1 is about the PATH, not the DATA.** Every endpoint runs the real
production chain; the artifact the operator injects may be captured or
synthetic, and both flow through identical steps.  No endpoint
re-invokes a step's primitive on its own — that was how a standalone
probe could silently drift from the production call it claimed to
mirror.

The mechanism is ``run(start, artifact, stop)``: inject an artifact at a
start step, let the chain propagate forward, and name the step whose
output you want back.  Each endpoint is one calibration use case and
DECLARES its own triple (see :data:`_CHAIN`); nothing is inferred from
the posted artifact's type.  The start request rides
:func:`~paramem.graph.phase_trace.start_at` and the stop request
:func:`~paramem.graph.phase_trace.stop_at`, so neither is threaded as a
parameter through the pipeline.

Endpoints running the extraction chain — ``POST /calibrate/{extract,
procedural,anonymize,enrich,plausibility}`` — share one request shape
(:class:`CalibrateChainRequest`) and one handler
(:func:`calibrate_chain`).  They reach the chain through
:class:`paramem.graph.extraction_pipeline.ExtractionPipeline`, the
single-topology chokepoint, on the process-wide ``ConsolidationLoop``
(lazy-built on first /consolidate or /calibrate call), so every flag the
production cycle applies is applied here too.  Endpoints that enter past
``local_extract`` inject the graph that stage would have produced;
endpoints whose own step sits inside a composite stage (``cloud_enrich``,
``deanon_plausibility``) declare the nearest start that exists and let
the chain produce the intermediate artifacts by running — which is why
the enrichment and plausibility use cases place a BILLED cloud call.

No call modifies weights or writes production data on disk.  Prompt
variants are resolved by name from ``paths.calibration/prompts/`` and
injected via :func:`~paramem.graph.prompts.prompt_overrides`; artifacts
land under ``paths.calibration/artifacts/``.

Returns a uniform shape:

  prompt_path, prompt_sha, prompt_template, raw_output, parsed,
  n_input_tokens, n_output_tokens, wall_clock_seconds, model,
  params_effective, vram_before, vram_after, artifact_dir.

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
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from paramem.graph.phase_trace import (
    PhaseRecord,
    extraction_trace,
    start_at,
    stop_at,
)
from paramem.graph.prompts import prompt_overrides
from paramem.graph.schema import SessionGraph
from paramem.server.gpu_lock import gpu_lock_sync
from paramem.server.session_buffer import SessionBuffer
from paramem.utils.artifacts import calibration_run, on_calibration_result, on_session_extracted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CalibrateParams(BaseModel):
    """Inference-time sampling overrides for a single calibration call.

    All fields default to ``None`` — the underlying call site uses its
    configured production default for every unset field.  ``seed`` only
    applies to local stages (Anthropic does not accept a seed parameter;
    Cloud stages report ``seed: null`` in ``params_effective``).  seed
    only affects output at temperature>0; at the default greedy
    temperature 0.0 it is a no-op.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int | None = None


class CalibrateChainRequest(BaseModel):
    """One request shape for every endpoint that runs the extraction chain.

    The endpoint — not this payload — declares where the chain is entered
    and which step's output comes back (see :data:`_CHAIN`).  What the
    caller supplies is the artifact and the run's own parameters.

    Attributes:
        transcript: The session text, turn-marked ``[user]``/``[assistant]``
            exactly as production receives it (see
            :func:`_require_turn_marked_transcript`).  Required for every
            use case: the chain consumes it at ``local_extract``, at
            ``anonymize`` (it is the text rewritten into the anonymized
            transcript that egresses), and at the de-anonymized
            plausibility judge.
        graph: A ``SessionGraph`` dict seeding a chain entered past
            ``local_extract`` — the graph that stage would have produced,
            typically taken verbatim from a prior ``/calibrate/extract``
            response.  Required exactly for the use cases whose
            declaration enters at ``anonymize``; rejected as unusable
            input (400) when absent there.
        stop_phase: Honoured only by use cases whose declaration leaves
            the stop open (``/calibrate/extract``): a name from
            :data:`~paramem.graph.phase_trace.PHASE_NAMES` after which the
            chain returns.  ``None`` runs to the end.  Every other use
            case fixes its own stop and ignores this.
        prompt_variants: ``{production basename: variant basename}`` —
            each variant is resolved from ``paths.calibration/prompts/``
            and injected via
            :func:`~paramem.graph.prompts.prompt_overrides`, so ANY step's
            prompt can be varied through the same one field rather than a
            per-endpoint filename knob.  A named variant that does not
            exist is a 400 before any inference runs.
    """

    transcript: str
    speaker_id: str
    graph: dict | None = None
    speaker_name: str | None = None
    source_type: str = Field(default="transcript", pattern="^(transcript|document)$")
    session_id: str = "calib"
    stop_phase: str | None = None
    prompt_variants: dict[str, str] = Field(default_factory=dict)
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

    ``prompt_variants`` carries the operator's prompt variants, resolved
    the same way every other calibration use case resolves them (see
    :func:`_resolve_prompt_variants`).
    """

    relations: list[dict] | None = None
    snapshot_path: str | None = None
    prompt_variants: dict[str, str] = Field(default_factory=dict)
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateNameRequest(BaseModel):
    """Run the production name extractor on an explicit turn list.

    ``turns`` is a list of ``{"role": str, "text": str}`` dicts — the same
    shape the production enrollment path receives from
    ``_run_enrollment_for_speaker``, and the artifact this use case
    injects.  When ``user_turns_only`` is ``True`` (default, mirrors
    production), only ``role == "user"`` turns are fed to the model;
    assistant turns are silently excluded so salutations like "Good
    evening, user" cannot be mis-classified as name introductions.

    ``prompt_variants`` carries the operator's prompt variants, resolved
    the same way every other calibration use case resolves them (see
    :func:`_resolve_prompt_variants`).
    """

    turns: list[dict]
    user_turns_only: bool = True
    prompt_variants: dict[str, str] = Field(default_factory=dict)
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


def _resolve_prompt_variants(state: dict, variants: dict[str, str]) -> dict[str, str]:
    """Read the operator's prompt variants into a
    :func:`~paramem.graph.prompts.prompt_overrides` mapping.

    ``variants`` maps a production prompt basename to the basename of the
    operator's variant of it; each variant is read from
    ``paths.calibration/prompts/``.  The returned
    ``{production basename: variant CONTENT}`` mapping is what
    ``prompt_overrides`` consumes, so ANY prompt the chain loads can be
    varied through this one mechanism — no per-endpoint filename knob, and
    no second resolution path alongside
    :func:`~paramem.graph.prompts._load_prompt`.

    Runs from a handler's ``guard`` closure — BEFORE any model call — so a
    typo'd variant name surfaces as HTTP 400 with zero inference cost.
    Resolution is STRICT: the variant must exist in the calibration prompt
    directory.  There is deliberately no fall-through to the shipped
    prompt of the same name, which would let a missing variant silently
    calibrate the production prompt instead.
    """
    base = state["config"].paths.calibration_prompts
    resolved: dict[str, str] = {}
    for production_name, variant_name in variants.items():
        path = base / variant_name
        if not path.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt variant not found: {path}. Calibration resolves variants "
                    f"strictly from the calibration prompt directory; it never falls "
                    f"back to the shipped {production_name!r}."
                ),
            )
        resolved[production_name] = path.read_text(encoding="utf-8").strip()
    return resolved


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


def _declared_step_unreached(state: dict, stage: str, phase: str, ran: list[str]) -> str:
    """Explain why the step a calibration reports on never ran.

    Every calibration endpoint promises the output of ONE named step. When
    the configured chain cannot reach that step — cloud egress refused, so
    the anonymize/enrich stages are skipped; an injected graph with no
    relations, so the anonymizer has nothing to anonymize; the
    normalization pass short-circuiting on its own floor — the honest
    answer is a refusal naming the gap, not a 200 whose provenance is
    silently empty.

    The cloud verdict is read from the SAME component and the SAME
    configuration object the chain's own gate uses
    (``paramem.graph.flows._session_egress_permitted`` feeds
    :func:`~paramem.cloud.admission.evaluate_cloud_egress` from the
    ``ExtractionConfig`` held by the pipeline this call runs on), so this
    reports the verdict rather than re-deriving it.
    """
    from paramem.cloud.admission import evaluate_cloud_egress

    detail = (
        f"The {stage!r} calibration reports the output of the {phase!r} step, "
        f"but that step did not run. Steps that ran: {ran or 'none'}."
    )
    loop = state.get("consolidation_loop")
    cfg = getattr(getattr(loop, "extraction", None), "config", None)
    if cfg is None:
        return detail
    verdict = evaluate_cloud_egress(
        cloud_enabled=cfg.cloud_enabled,
        provider=cfg.enrichment_provider,
        model=cfg.enrichment_provider_model,
        endpoint=cfg.enrichment_provider_endpoint or None,
    )
    if verdict.permitted:
        return detail + " Cloud egress is permitted, so the gap is upstream of it."
    return (
        detail
        + " Cloud egress is refused ("
        + "; ".join(verdict.gaps)
        + "), and every step from 'anonymize' onward sits behind that gate."
    )


def _run_calibration(
    *,
    stage: str,
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
    5. ``dispatch()`` is called bare: every calibration use case runs a
       production path, and every production path opens its own named
       phases onto this outer trace. Nothing here synthesises a phase — a
       calibration-only phase record is exactly the divergence this
       substrate exists to prevent.
    6. Refuses (400) when no phase record for ``input_prompt_phase`` exists:
       the endpoint promised that step's output, and a configuration or an
       input that never reaches it must be reported, not papered over with
       an envelope carrying empty provenance. See
       :func:`_declared_step_unreached`.
    7. Assembles the uniform 11-key response (+ ``phases``) directly from
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
        guard: Zero-arg callable raising :class:`~fastapi.HTTPException`
            (400) for any invalid input. Called before the trace opens.
        dispatch: Zero-arg callable that runs the step and returns
            ``(raw_output, parsed)``.
        input_prompt_phase: Which phase record's non-system prompt
            template feeds ``n_input_tokens`` (see
            :func:`_provenance_from_records`).
        state: The live server state dict.
        params: The request's :class:`CalibrateParams`.
        supports_seed: ``True`` for local stages, ``False`` for cloud
            stages (mirrors the existing ``_effective_params`` convention).

    Returns:
        The uniform calibration response dict.
    """
    _preflight(state)
    guard()
    _ensure_calibration_loop(state)
    # Everything this run produces — the response below, and any artifact a
    # production hook emits while the run executes (the graph tier's
    # normalization pass writes its raw outputs through the same artifact
    # hooks) — lands in the run's own directory, whether or not the production
    # debug switch is on. With debug on, the debug tree receives it too.
    run_dir = state["config"].paths.calibration_artifacts / f"{stage}_{int(time.time())}"
    with calibration_run(run_dir):
        with extraction_trace() as trace:
            with _measured_local_call() as m:
                # The production path (extract_graph, the graph-tier pass,
                # the name extractor) opens its own named phases onto this
                # same outer trace via the extraction_trace() nesting no-op
                # — nothing to wrap here.
                raw_output, parsed = dispatch()
        records = trace.records
        if input_prompt_phase not in {r.name for r in records}:
            raise HTTPException(
                status_code=400,
                detail=_declared_step_unreached(
                    state, stage, input_prompt_phase, [r.name for r in records]
                ),
            )
        prompts, input_prompt_text = _provenance_from_records(records, input_prompt_phase)

        tokenizer = state.get("tokenizer")
        n_in = _count_tokens(tokenizer, input_prompt_text) if tokenizer else -1
        count_str = raw_output if isinstance(raw_output, str) else ""
        n_out = _count_tokens(tokenizer, count_str) if (tokenizer and count_str) else -1
        model_id = getattr(state.get("model_id"), "name", state.get("model_id", "unknown"))

        response = {
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
            # Where this run's artifacts live. The run — not the caller —
            # owns its record: the response written below, plus anything a
            # production hook emitted while it executed. A client reads
            # them from here instead of keeping its own copy.
            "artifact_dir": str(run_dir),
        }
        on_calibration_result(response)
    return response


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


@dataclass(frozen=True)
class _ChainDeclaration:
    """What one calibration use case does to the extraction chain.

    The endpoint declares; nothing is inferred from the posted artifact.
    This is the whole of ``run(start, artifact, stop)`` for one use case.

    Attributes:
        injects: Which artifact the caller supplies — ``"transcript"``
            (the chain is entered at its first step, which consumes the
            transcript the request already carries) or ``"graph"`` (a
            ``SessionGraph`` seeds a chain entered further along).
        start: The :data:`~paramem.graph.phase_trace.PHASE_NAMES` member
            whose stage the chain is entered at, opened via
            :func:`~paramem.graph.phase_trace.start_at`.
        stop: The step whose output this use case exists to inspect,
            opened via :func:`~paramem.graph.phase_trace.stop_at`.
            ``None`` for the use case that leaves the stop to the
            operator (``/calibrate/extract``, which exists precisely to
            inspect any point of a transcript-fed run).
        entry: The :class:`~paramem.graph.extraction_pipeline.ExtractionPipeline`
            method that runs this chain — the single-topology chokepoint,
            never an extractor primitive.
    """

    injects: str
    start: str
    stop: str | None
    entry: str


# The declaration per calibration use case.  A use case whose own step sits
# inside a composite stage (``cloud_enrich`` and ``deanon_plausibility`` are
# opened by the ``enrich``/``deanonymize`` stage bodies, not by stages of
# their own) declares the nearest start that exists — ``anonymize`` — and
# lets the chain produce the intermediate artifacts by running.  Those runs
# therefore place a real cloud call, which is the point: the de-anonymized
# judge production runs is only reachable downstream of one.
_CHAIN: dict[str, _ChainDeclaration] = {
    "extract": _ChainDeclaration(
        injects="transcript", start="local_extract", stop=None, entry="run"
    ),
    "procedural": _ChainDeclaration(
        injects="transcript",
        start="procedural_extract",
        stop="procedural_extract",
        entry="run_procedural",
    ),
    "anonymize": _ChainDeclaration(
        injects="graph", start="anonymize", stop="anonymize", entry="run"
    ),
    "enrich": _ChainDeclaration(
        injects="graph", start="anonymize", stop="cloud_enrich", entry="run"
    ),
    "plausibility": _ChainDeclaration(
        injects="graph", start="anonymize", stop="deanon_plausibility", entry="run"
    ),
}


def calibrate_chain(state: dict, use_case: str, req: CalibrateChainRequest) -> dict[str, Any]:
    """Run the production extraction chain for one calibration use case.

    The one handler behind every chain endpoint.  It reads that use
    case's declaration (:data:`_CHAIN`), opens the operator's prompt
    variants, enters the chain where the declaration says, and returns the
    declared step's output — it never calls a step's primitive itself, so
    a calibration run cannot diverge from the production call it reports
    on.

    ``guard`` rejects unusable input (turn-marking, a missing or
    unparseable graph seed, a named prompt variant that does not exist)
    before any inference runs; ``dispatch`` routes through
    :meth:`ExtractionPipeline.run` / :meth:`~ExtractionPipeline.run_procedural`
    on the process-wide ``ConsolidationLoop``, so calibration shares the
    exact instance the production /consolidate cycle uses — same model,
    same config, same flags.

    Prompt provenance and ``n_input_tokens`` come from the phase record
    the chain itself opened for the inspected step (see
    :func:`_provenance_from_records`) — never a hand-built ``prompts``
    literal.
    """
    decl = _CHAIN[use_case]
    # The operator's stop is honoured only where the declaration leaves one
    # open; every other use case fixes its own.
    stop = decl.stop if decl.stop is not None else req.stop_phase
    # The step this use case exists to inspect: whichever step the run
    # actually stops at — the declaration's, or the operator's where the
    # declaration leaves it open — falling back to the entry step when the
    # run has no stop at all and walks to the end.
    focus = stop or decl.start
    resolved: dict[str, Any] = {}

    def guard() -> None:
        _require_turn_marked_transcript(req.transcript)
        if not req.speaker_id:
            raise HTTPException(
                status_code=400,
                detail="speaker_id is required (no empty-string default).",
            )
        if decl.injects == "graph":
            if req.graph is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"This calibration enters the chain at {decl.start!r}, which "
                        f"consumes the graph 'local_extract' would have produced. "
                        f"Supply it as 'graph' (typically a prior /calibrate/extract "
                        f"response's parsed graph)."
                    ),
                )
            try:
                resolved["seed"] = SessionGraph.model_validate(req.graph)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid SessionGraph payload: {exc}",
                ) from exc
        resolved["overrides"] = _resolve_prompt_variants(state, req.prompt_variants)

    def dispatch() -> tuple[Any, dict]:
        loop = _ensure_calibration_loop(state)
        kwargs: dict[str, Any] = {
            "speaker_id": req.speaker_id,
            "speaker_name": req.speaker_name,
            "source_type": req.source_type,
            "seed": req.params.seed,
        }
        if decl.entry == "run":
            # Sampling overrides reach the chain through extract_graph's
            # signature; run_procedural's does not carry them.
            if req.params.max_tokens is not None:
                kwargs["max_tokens"] = req.params.max_tokens
            if req.params.temperature is not None:
                kwargs["temperature"] = req.params.temperature

        with (
            prompt_overrides(resolved["overrides"]),
            start_at(decl.start, resolved.get("seed")),
            stop_at(stop),
        ):
            graph = getattr(loop.extraction, decl.entry)(
                req.transcript,
                req.session_id,
                **kwargs,
            )
        # Symmetric to ConsolidationLoop.extract_session: the caller that turned
        # a transcript into a session graph persists it as the per-session
        # snapshot.  The calibration_run scope _run_calibration opened routes it
        # into this run's own directory (the production debug tree, too, when
        # debug is on).  Mid-chain endpoints (injects="graph") inject a graph and
        # run a sub-step — not a session extraction — so they write no snapshot.
        if decl.injects == "transcript":
            on_session_extracted(
                graph,
                req.session_id,
                "procedural_graph" if decl.entry == "run_procedural" else "graph",
            )
        parsed = graph.model_dump(mode="json") if hasattr(graph, "model_dump") else {}

        # The inspected step's own raw output, surfaced at the top level so
        # a prompt diff needs no traversal of the phases list.
        from paramem.graph.phase_trace import get_phases

        record = next(
            (r.to_dict() for r in get_phases(graph) if r.name == focus),
            None,
        )
        return (record.raw_output if record else "") or "", parsed

    response = _run_calibration(
        stage=use_case,
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase=focus,
        state=state,
        params=req.params,
        # Every chain run threads the seed into the local steps it walks,
        # whatever step it is inspecting — the local/cloud split that used
        # to gate this belonged to the standalone probes.
        supports_seed=True,
    )
    return response


def calibrate_normalize(state: dict, req: CalibrateNormalizeRequest) -> dict[str, Any]:
    """Run the production predicate-normalization pass on injected relations.

    Normalization is a single-step chain whose artifact is a relation set.
    ``dispatch`` seeds a throwaway :class:`~paramem.graph.merger.GraphMerger`
    with that set and hands it to the SAME
    :class:`~paramem.training.graph_tier.GraphTierRefiner` the consolidation
    cycle builds (``ConsolidationLoop.build_tier_refiner`` is the one
    construction site), so the operator sees the production engine selection
    — cloud when egress is permitted, local otherwise — and the production
    survivor rule (highest ``reinforcement_count``, not first-in-cluster).

    Nothing here re-derives what the pass would have done: the retirements
    reported are the ones the pass actually applied, to a graph built from
    the injected relations and discarded when the call returns.  The live
    merger is never touched.

    ``guard`` resolves the injected relations — supplied inline, or read
    from a NetworkX node-link snapshot on the server filesystem — and the
    operator's prompt variants, both before any model call.
    """
    from paramem.graph.merger import GraphMerger
    from paramem.graph.schema import Relation

    resolved: dict[str, Any] = {}

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
        resolved["overrides"] = _resolve_prompt_variants(state, req.prompt_variants)

        if has_relations:
            resolved["relations"] = req.relations
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
        loop = _ensure_calibration_loop(state)
        merger = GraphMerger(model=state.get("model"), tokenizer=state.get("tokenizer"))
        # The pass reads subject/predicate/object and the edge bookkeeping
        # the merger itself stamps; ``relation_type``/``speaker_id`` are
        # required by the schema but never consulted by it, so an injected
        # triple that omits them gets a structural placeholder rather than
        # forcing the operator to supply provenance the calibration does
        # not use.
        merger.merge_relations(
            [
                Relation(
                    subject=str(rel.get("subject", "")),
                    predicate=str(rel.get("predicate", "")),
                    object=str(rel.get("object", "")),
                    relation_type=rel.get("relation_type", "factual"),
                    speaker_id=rel.get("speaker_id", "speaker0"),
                )
                for rel in resolved["relations"]
            ],
            session_id="__calibration_normalize__",
            log_label="calibration",
        )
        before = merger.get_all_triples()
        # extraction_trace() re-entry is a no-op that yields the scope
        # _run_calibration already opened, which is where the pass's own
        # nested scope lands the ``normalize`` phase record.
        with extraction_trace() as trace, prompt_overrides(resolved["overrides"]):
            diagnostics = loop.build_tier_refiner(merger).run_normalization()
        after = merger.get_all_triples()
        retired = [list(triple) for triple in set(before) - set(after)]

        record = next((r for r in trace.records if r.name == "normalize"), None)
        parsed: dict[str, Any] = {
            "surviving_relations": [list(triple) for triple in after],
            "retired_relations": retired,
            "input_count": len(resolved["relations"]),
            "surviving_count": len(after),
            **diagnostics,
        }
        return (record.raw_output if record else "") or "", parsed

    return _run_calibration(
        stage="normalize",
        guard=guard,
        dispatch=dispatch,
        input_prompt_phase="normalize",
        state=state,
        params=req.params,
        supports_seed=True,
    )


def calibrate_name(state: dict, req: CalibrateNameRequest) -> dict[str, Any]:
    """Run the production name extractor on an explicit turn list.

    Enrollment is a single-step chain: the artifact injected at
    ``name_extract`` is the turn list, and that step is also the one whose
    output comes back.  ``dispatch`` calls
    :func:`~paramem.graph.name_extraction.extract_name_via_llm` — the same
    function, on the same base weights, that
    ``_run_enrollment_for_speaker`` calls in production, and which opens
    the ``name_extract`` phase itself.  Nothing here re-implements the
    post-filter or synthesises a phase record: ``prompts``,
    ``raw_output`` and ``n_input_tokens`` all come from the phase the
    primitive opened.

    ``user_turns_only`` mirrors the production default (``True``) — only
    user turns reach the model; set to ``False`` to include assistant turns
    and reproduce the original (buggy) context-scoping behaviour for
    comparative testing.
    """
    from paramem.graph.name_extraction import extract_name_via_llm
    from paramem.models.loader import base_model_inference

    model = state.get("model")
    tokenizer = state.get("tokenizer")
    inference_params = {
        "temperature": req.params.temperature,
        "seed": req.params.seed,
        "max_tokens": req.params.max_tokens,
    }
    resolved: dict[str, Any] = {}

    def guard() -> None:
        resolved["overrides"] = _resolve_prompt_variants(state, req.prompt_variants)

    def dispatch() -> tuple[Any, dict]:
        with prompt_overrides(resolved["overrides"]), base_model_inference(model):
            extracted, raw_output = extract_name_via_llm(
                req.turns,
                model,
                tokenizer,
                user_turns_only=req.user_turns_only,
                params=inference_params,
            )
        return raw_output, {"name": extracted}

    return _run_calibration(
        stage="name",
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
    via a scoped torch.Generator) from cloud stages (where Anthropic's API
    accepts no seed parameter and the field is silently dropped).  The
    response uses this to inform the operator which fields actually
    landed.

    seed threads through all three local stages (extract, anonymize,
    plausibility) via the helpers' ``seed`` parameter forwarded to
    ``generate_answer``.  top_p / top_k are not yet threaded for these
    stages (documented gap; cloud stages follow the Anthropic API which
    omits them).  The field is reported as-requested for transparency.
    """
    out: dict = {}
    for f in ("temperature", "top_p", "top_k", "max_tokens"):
        out[f] = getattr(params, f)
    out["seed"] = params.seed if supports_seed else None
    return out
