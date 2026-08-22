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
(:class:`CalibrateChainRequest`) and one validate/dispatch pair
(:func:`validate_chain` / :func:`dispatch_chain`).  They reach the chain
through :class:`paramem.graph.extraction_pipeline.ExtractionPipeline`, the
single-topology chokepoint, on the process-wide ``ConsolidationLoop``
(lazy-built on first dispatch), so every flag the production cycle applies
is applied here too.  Endpoints that enter past ``local_extract`` inject
the graph that stage would have produced; endpoints whose own step sits
inside a composite stage (``cloud_enrich``, ``deanon_plausibility``)
declare the nearest start that exists and let the chain produce the
intermediate artifacts by running — which is why the enrichment and
plausibility use cases place a BILLED cloud call.

The module also hosts standalone validate/dispatch pairs outside the
chain — :func:`validate_normalize` / :func:`dispatch_normalize`
(``POST /calibrate/normalize``), :func:`validate_anonymize_facts` /
:func:`dispatch_anonymize_facts` (``POST /calibrate/anonymize_facts``),
:func:`validate_name` / :func:`dispatch_name` (``POST /calibrate/name``),
:func:`validate_respond` / :func:`dispatch_respond`
(``POST /calibrate/respond``), and the pending-session probe
(``POST /calibrate/extract_pending``, dispatched from the server layer
since its step is :func:`~paramem.server.app._extract_pending_sessions`)
— each with its own request shape; see each function's docstring.

No call modifies weights or writes production data on disk.  Prompt
variants are resolved by name from ``paths.calibration/prompts/`` and
injected via :func:`~paramem.graph.prompts.prompt_overrides`; artifacts
land under ``paths.calibration/artifacts/``.

**Non-blocking.**  Every ``/calibrate/*`` route is a consolidation
dispatch: the same guards, the same ``consolidating`` mutex, the same
executor hop, GPU lock, cooldown gate, and terminal a production fold
gets.  A route answers HTTP 200 ``{status, action[, run_id, artifact_dir]}``
at once; the run itself executes on an executor thread and writes its
full result to ``<artifact_dir>/response.json`` — this module returns no
result body over HTTP.  A route's own boundary work (:func:`preflight`,
per-stage ``validate_*``, prompt-variant resolution) runs on the event
loop, before dispatch; :func:`run_stage` is what the executor calls.

Returns a uniform response-file shape:

  stage, prompts, raw_output, parsed, n_input_tokens, n_output_tokens,
  wall_clock_seconds, model, params_effective, vram_before, vram_after,
  phases, artifact_dir, run_id, unreached_step.

Gating: every endpoint short-circuits with 404 when the server config's
``calibrate_endpoint_enabled`` flag is False.  Default is False —
calibration is opt-in via ``configs/server.yaml``, never live in
production.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException
from pydantic import BaseModel, Field

from paramem.graph.phase_trace import (
    PHASE_NAMES,
    PhaseRecord,
    extraction_trace,
    phase_trace,
    start_at,
    stop_at,
)
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema import SessionGraph
from paramem.server import lang_id
from paramem.server.session_buffer import SessionBuffer
from paramem.utils.artifacts import on_session_extracted
from paramem.utils.tokens import estimate_tokens

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
    :func:`resolve_prompt_variants`).
    """

    relations: list[dict] | None = None
    snapshot_path: str | None = None
    prompt_variants: dict[str, str] = Field(default_factory=dict)
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateAnonymizeFactsRequest(BaseModel):
    """Run the graph-tier anonymize step — the facts-only shape
    :func:`~paramem.training.graph_enrich.enrich_graph` sends per chunk —
    on an explicit fact list or a graph snapshot.

    Distinct from ``/calibrate/anonymize`` (session tier: a transcript,
    chained through :class:`~paramem.graph.extraction_pipeline.ExtractionPipeline`).
    The graph tier has no transcript at all; ``anonymize()`` is called
    directly with ``transcript=""``, so this request carries a fact list
    instead — mirroring :class:`CalibrateNormalizeRequest`'s two
    equally-valid artifact sources:

    * ``facts`` — flat list of fact dicts (``subject``, ``predicate``,
      ``object``, and optionally ``relation_type``/``speaker_id`` —
      the shape :func:`~paramem.training.graph_enrich.serialize_subgraph_triples`
      produces), supplied directly by the caller.
    * ``snapshot_path`` — path to a NetworkX node-link
      ``graph_merged_snapshot.json`` on the server filesystem, read via
      the same :func:`_relations_from_snapshot` reader
      :class:`CalibrateNormalizeRequest` uses.

    Exactly one of ``facts`` or ``snapshot_path`` must be provided;
    supplying neither or both raises HTTP 400.

    ``identity_domain`` (the reconciliation domain :func:`~paramem.cloud.
    anonymize.anonymize` reconciles the model's mapping against) is
    derived server-side from the resolved facts' own subject/object
    endpoints — mirroring a production chunk's node list — never supplied
    by the caller.  ``scrub`` and ``token_envelope`` are read from the
    SAME ``ExtractionConfig`` production reads (never a request
    override), so this calibrates against the operator's actual
    configuration, not a synthetic one.

    ``prompt_variants`` carries the operator's prompt variants, resolved
    the same way every other calibration use case resolves them (see
    :func:`resolve_prompt_variants`); the one basename this stage loads
    is ``anonymization_facts.txt``.
    """

    facts: list[dict] | None = None
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
    :func:`resolve_prompt_variants`).
    """

    turns: list[dict]
    user_turns_only: bool = True
    prompt_variants: dict[str, str] = Field(default_factory=dict)
    params: CalibrateParams = Field(default_factory=CalibrateParams)


class CalibrateRespondRequest(BaseModel):
    """Run one serving turn through the production chat dispatch
    (:func:`~paramem.server.inference.handle_chat`) for an enrolled speaker.

    ``text`` is a bare utterance — not a turn-marked transcript; the
    serving path never expects the ``[user]``/``[assistant]`` markers
    :func:`_require_turn_marked_transcript` requires from the extraction
    chain endpoints. ``conversation_id`` selects which stored history
    :meth:`~paramem.server.session_buffer.SessionBuffer.get_conversation_turns`
    reads back; the default reads back empty (a fresh conversation), same as
    every other calibration use case's default id.

    Deliberately carries **no sampling-parameter field**. The serving
    generate hardcodes ``temperature=0.0`` and takes its token cap from
    ``config.inference.max_response_tokens`` (there is no production caller
    of either as a per-request override) — offering a ``params`` field here
    that the call silently ignored would be exactly the ``top_p``/``top_k``
    echo-without-effect pattern this module already documents as a defect
    for the chain endpoints, reproduced on purpose.

    ``prompt_variants`` carries the operator's prompt variants, resolved
    the same way every other calibration use case resolves them (see
    :func:`resolve_prompt_variants`).
    """

    text: str
    speaker_id: str
    conversation_id: str = "calib-respond"
    prompt_variants: dict[str, str] = Field(default_factory=dict)


class CalibrateExtractPendingRequest(BaseModel):
    """Body for ``POST /calibrate/extract_pending``.

    The run's artifact is the pending NAMED session set — the same set a
    fold takes — so this payload carries only what an operator can vary
    about the run itself.

    Attributes:
        prompt_variants: ``{production basename: variant basename}``,
            resolved from ``paths.calibration_prompts`` and injected via
            :func:`~paramem.graph.prompts.prompt_overrides` — the family's
            one prompt field, identical in shape and resolution to every
            other calibration request.  Supplied by the OPERATOR; the
            driver script expands ``--prompt-prefix`` into it
            (``scripts/dev/calibrate_prompts.py::_variants``).
        params: Sampling overrides for this run's local calls, same
            semantics as every other chain request.
    """

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


def resolve_prompt_variants(state: dict, variants: dict[str, str]) -> dict[str, str]:
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


def _relations_from_snapshot(snapshot_path: str) -> list[dict]:
    """Load a NetworkX node-link ``graph_merged_snapshot.json`` into flat
    ``{subject, predicate, object, relation_type}`` dicts, plus
    ``speaker_id`` only when the edge carries one.

    THE one snapshot-to-relations reader — shared by
    :func:`validate_normalize` and :func:`validate_anonymize_facts` so a
    future snapshot-format change updates one place, not two independently
    hand-rolled parses of the same file shape.

    NetworkX node-link format: ``{"nodes": [...], "links": [...]}`` where
    each link is ``{source, target, key, ...edge_data...}``. Edges missing
    a ``predicate`` key are skipped (non-relation edges, if any). An edge
    with no ``speaker_id`` (or an empty one) omits the key entirely rather
    than emitting ``""`` — :func:`dispatch_normalize`'s structural
    ``rel.get("speaker_id", "speaker0")`` placeholder then applies, the
    same as an inline-supplied relation that omits the field.
    :func:`dispatch_anonymize_facts` never indexes a fact's
    ``speaker_id``, so omitting the key is safe there too. Raises
    :class:`~fastapi.HTTPException` (400) when the path does not exist or
    is not valid JSON — this is a guard-time check, called before any
    model call.
    """
    import json as _json
    from pathlib import Path as _Path

    snap_path = _Path(snapshot_path)
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
    links = snap.get("links", snap.get("edges", []))
    relations: list[dict] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        pred = link.get("predicate")
        if not pred:
            continue
        relation: dict[str, Any] = {
            "subject": str(link.get("source", "")),
            "predicate": str(pred),
            "object": str(link.get("target", "")),
            "relation_type": str(link.get("relation_type", "factual")),
        }
        speaker_id = link.get("speaker_id")
        if speaker_id:
            relation["speaker_id"] = str(speaker_id)
        relations.append(relation)
    return relations


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


def preflight(state: dict) -> None:
    """Raise the appropriate HTTP exception when the server is not in a
    state that can serve a calibration call.

    Runs on the event loop, before dispatch — every arm checks a handle
    directly, never ``state["mode"]`` (which has multiple assignment sites
    and can lag them).  Whether a real consolidation cycle is currently
    running is NOT checked here: a calibration call now runs under the
    same dispatch envelope as a fold, so a busy server answers 200
    ``deferred_already_running`` from the arbitrator instead of a 503 here.

    * 404 when the calibrate flag is off — the endpoint shouldn't exist
      from the client's perspective.
    * 503 ``model_not_loaded`` when the local model or tokenizer isn't
      resident (cloud-only mode, defer-model boot).
    * 503 ``store_unavailable`` when the memory store hasn't been
      constructed yet (or is quarantined) — this is also what keeps a
      calibration dispatch from being the first construction of the
      process-lifetime loop singleton with no store override, while a
      pending event's own resume is waiting to pass one.
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
    if state.get("model") is None or state.get("tokenizer") is None:
        raise HTTPException(
            status_code=503,
            detail="Local model not loaded (cloud-only mode or "
            "defer-model boot). Calibration requires a local model.",
        )
    if state.get("memory_store") is None:
        raise HTTPException(
            status_code=503,
            detail="Memory store not available (not yet constructed, or "
            "quarantined). Calibration requires a live memory store.",
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
            (one :class:`ExtractionTrace` per :func:`run_stage` call).
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


@dataclass(frozen=True)
class CalibrationRunSpec:
    """One validated calibration run, ready to execute.

    Built on the event loop by the route handler; consumed on the executor
    thread.  Every field is resolved — no request object and no unvalidated
    string crosses the thread boundary.

    Attributes:
        stage: The route's own path segment (``"extract"``, ``"normalize"``,
            ``"extract_pending"``, …) — the response's ``stage`` label.
        route_path: The producing route (``"/calibrate/extract"``).
        run_id: This run's stamp, minted at the boundary and already
            returned to the caller.
        artifact_dir: This run's directory, already returned to the caller.
        dispatch: Zero-arg callable running the step and returning
            ``(raw_output, parsed)`` — the per-stage dispatch closure, with
            its validated input already bound.
        input_prompt_phase: Which phase record's non-system prompt feeds
            ``n_input_tokens`` and the provenance block.
        supports_seed: Whether ``params_effective["seed"]`` echoes the
            request's seed.
        params: The request's :class:`CalibrateParams`.
        overrides: ``{production basename: variant CONTENT}`` for
            :func:`~paramem.graph.prompts.prompt_overrides`.
        evicts_voice: Whether this run's own artifact is document-shaped.
    """

    stage: str
    route_path: str
    run_id: str
    artifact_dir: Path
    dispatch: "Callable[[], tuple[Any, Any]]"
    input_prompt_phase: str
    supports_seed: bool
    params: CalibrateParams
    overrides: dict[str, str] = field(default_factory=dict)
    evicts_voice: bool = False


def run_stage(spec: CalibrationRunSpec, state: dict) -> dict[str, Any]:
    """Run one calibration step and assemble its response — the shared
    assembler behind every calibration route, called by
    :func:`~paramem.server.app._run_calibration_sync` inside the run's
    artifact scopes (the one owner of that scope for a real dispatch).

    Does exactly what only it can do: open ``extraction_trace()`` around
    the step, derive the prompt provenance and ``n_input_tokens`` from the
    phase records the chain itself opened
    (:func:`_provenance_from_records`) rather than a hand-built literal,
    count output tokens, read ``model_id`` and ``params_effective``, and
    record the run's own measurement.  The preflight gates, the input
    guard, the artifact scope, and the GPU lock are the envelope's own —
    :func:`~paramem.server.app._run_calibration_sync` owns those, not this
    function.

    The measurement is retained and is lock-free: ``wall_clock_seconds``
    (``time.perf_counter`` around ``spec.dispatch()``) and
    ``vram_before`` / ``vram_after`` are captured here, inside the
    envelope's lock rather than around it, so the recorded value is the
    step's own cost — not time spent waiting for the GPU.

    Args:
        spec: The validated run.
        state: The live server state dict — read for ``tokenizer`` (token
            counting), ``model_config.model_id``, and (on an unreached
            declared step) the loop's extraction config for the cloud-egress
            verdict (:func:`_declared_step_unreached`).

    Returns:
        The response dict — the same field set every calibration route has
        always returned (``stage``, ``prompts``, ``raw_output``, ``parsed``,
        ``n_input_tokens``, ``n_output_tokens``, ``wall_clock_seconds``,
        ``model``, ``params_effective``, ``vram_before``, ``vram_after``,
        ``phases``, ``artifact_dir``) plus ``run_id`` and ``unreached_step``.
        It is written to disk by the caller rather than returned over HTTP.
    """
    with extraction_trace() as trace:
        vram_before = _vram_block()
        t0 = time.perf_counter()
        # The production path (extract_graph, the graph-tier pass, the name
        # extractor, handle_chat) opens its own named phases onto this same
        # outer trace via the extraction_trace() nesting no-op — nothing to
        # wrap here.  Nothing here synthesises a phase.
        raw_output, parsed = spec.dispatch()
        wall_clock_seconds = time.perf_counter() - t0
        vram_after = _vram_block()

    records = trace.records
    ran = [r.name for r in records]
    unreached_step: dict[str, Any] | None = None
    if spec.input_prompt_phase not in set(ran):
        # The declared step never ran.  Post-200 there is no response left
        # to fail with a 400 — the run still completes and writes every
        # phase that DID run; the gap is reported as data instead.
        unreached_step = {
            "declared_phase": spec.input_prompt_phase,
            "phases_ran": ran,
            "detail": _declared_step_unreached(state, spec.stage, spec.input_prompt_phase, ran),
        }
    prompts, input_prompt_text = _provenance_from_records(records, spec.input_prompt_phase)

    tokenizer = state.get("tokenizer")
    n_in = estimate_tokens(input_prompt_text, tokenizer) if tokenizer else -1
    count_str = raw_output if isinstance(raw_output, str) else ""
    n_out = estimate_tokens(count_str, tokenizer) if (tokenizer and count_str) else -1
    model_id = state["config"].model_config.model_id

    return {
        "stage": spec.stage,
        "prompts": prompts,
        "raw_output": raw_output,
        "parsed": parsed,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "wall_clock_seconds": wall_clock_seconds,
        "model": model_id,
        "params_effective": _effective_params(spec.params, supports_seed=spec.supports_seed),
        "vram_before": vram_before,
        "vram_after": vram_after,
        "phases": [r.to_dict() for r in records],
        # Where this run's artifacts live. The run — not the caller — owns
        # its record: the response written below, plus anything a
        # production hook emitted while it executed. A client reads them
        # from here instead of keeping its own copy.
        "artifact_dir": str(spec.artifact_dir),
        "run_id": spec.run_id,
        "unreached_step": unreached_step,
    }


# ---------------------------------------------------------------------------
# Stage handlers — invoked from the registered FastAPI routes in app.py
# ---------------------------------------------------------------------------


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


def validate_chain(state: dict, use_case: str, req: CalibrateChainRequest) -> dict[str, Any]:
    """Validate one chain-endpoint request — paired with :func:`dispatch_chain`
    as the use case's validate/dispatch split.

    Runs on the event loop, before dispatch: rejects unusable input
    (turn-marking, a missing or unparseable graph seed, a named prompt
    variant that does not exist) BEFORE any model call, zero inference
    cost.  Resolves the use case's declaration (:data:`_CHAIN`), the stop
    step, and the operator's prompt overrides into one dict
    :func:`dispatch_chain` consumes.

    Returns:
        The resolved dict: ``decl``, ``stop``, ``focus`` (the inspected
        step), ``overrides``, and (only when ``decl.injects == "graph"``)
        ``seed`` (the validated :class:`~paramem.graph.schema.SessionGraph`).
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
    resolved: dict[str, Any] = {"decl": decl, "stop": stop, "focus": focus}

    # The SAME source (PHASE_NAMES) stop_at validates again downstream as a
    # library precondition. Duplicating the membership test is deliberate:
    # one is a request-input check, the other guards the pipeline call
    # regardless of caller.
    if stop is not None and stop not in PHASE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"stop_phase {stop!r} is not a valid phase name. Valid: {list(PHASE_NAMES)}",
        )
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
    resolved["overrides"] = resolve_prompt_variants(state, req.prompt_variants)
    # Document-shaped only for the transcript-injecting use cases: a
    # mid-chain endpoint (injects="graph") never runs the dense-chunk
    # extraction regime the eviction exists for.
    resolved["evicts_voice"] = decl.injects == "transcript" and req.source_type == "document"
    return resolved


def dispatch_chain(
    state: dict, use_case: str, req: CalibrateChainRequest, resolved: dict[str, Any]
) -> tuple[Any, dict]:
    """Run one calibration use case's chain step — paired with
    :func:`validate_chain`, which validates the request and resolves the
    use case's declaration before this runs.

    Routes through :meth:`ExtractionPipeline.run` /
    :meth:`~ExtractionPipeline.run_procedural` on the process-wide
    ``ConsolidationLoop``, so calibration shares the exact instance the
    production consolidation dispatch uses — same model, same config, same
    flags.  Prompt provenance and ``n_input_tokens`` come from the phase
    record the chain itself opened for the inspected step (see
    :func:`_provenance_from_records`) — never a hand-built ``prompts``
    literal.

    Args:
        state: The live server state dict.
        use_case: The route's use case name (``_CHAIN`` key).
        req: The validated request.
        resolved: :func:`validate_chain`'s return value.
    """
    from paramem.server.consolidation import get_or_create_consolidation_loop

    decl = resolved["decl"]
    stop = resolved["stop"]
    focus = resolved["focus"]
    loop = get_or_create_consolidation_loop(state)
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
    # snapshot.  The calibration_run scope the envelope opened routes it
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
        (r for r in get_phases(graph) if r.name == focus),
        None,
    )
    return (record.raw_output if record else "") or "", parsed


def validate_normalize(state: dict, req: CalibrateNormalizeRequest) -> dict[str, Any]:
    """Resolve the injected relations and prompt overrides for
    :func:`dispatch_normalize` — paired with it as the ``normalize`` stage's
    validate/dispatch split.

    Resolves the injected relations — supplied inline, or read from a
    NetworkX node-link snapshot on the server filesystem — and the
    operator's prompt variants, both before any model call.
    """
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
    resolved: dict[str, Any] = {"overrides": resolve_prompt_variants(state, req.prompt_variants)}
    resolved["relations"] = (
        req.relations if has_relations else _relations_from_snapshot(req.snapshot_path)  # type: ignore[arg-type]
    )
    return resolved


def dispatch_normalize(
    state: dict, req: CalibrateNormalizeRequest, resolved: dict[str, Any]
) -> tuple[Any, dict]:
    """Run the production predicate-normalization pass on the resolved
    relations — paired with :func:`validate_normalize`, which resolves
    *resolved* before this runs.

    Seeds a throwaway :class:`~paramem.graph.merger.GraphMerger` with the
    resolved relation set and hands it to the SAME
    :class:`~paramem.training.graph_tier.GraphTierRefiner` the consolidation
    cycle builds (``ConsolidationLoop.build_tier_refiner`` is the one
    construction site), so the operator sees the production engine selection
    — cloud when egress is permitted, local otherwise — and the production
    survivor rule (highest ``reinforcement_count``, not first-in-cluster).

    Nothing here re-derives what the pass would have done: the retirements
    reported are the ones the pass actually applied, to a graph built from
    the injected relations and discarded when the call returns.  The live
    merger is never touched.
    """
    from paramem.graph.merger import GraphMerger
    from paramem.graph.schema import Relation
    from paramem.server.consolidation import get_or_create_consolidation_loop

    loop = get_or_create_consolidation_loop(state)
    merger = GraphMerger(model=state.get("model"), tokenizer=state.get("tokenizer"))
    # The pass reads subject/predicate/object and the edge bookkeeping
    # the merger itself stamps; ``relation_type``/``speaker_id`` are
    # required by the schema but never consulted by it, so an injected
    # triple that omits them — or supplies an explicit empty
    # ``speaker_id``, inline or snapshot-sourced — gets a structural
    # placeholder rather than forcing the operator to supply provenance
    # the calibration does not use.
    merger.merge_relations(
        [
            Relation(
                subject=str(rel.get("subject", "")),
                predicate=str(rel.get("predicate", "")),
                object=str(rel.get("object", "")),
                relation_type=rel.get("relation_type", "factual"),
                speaker_id=rel.get("speaker_id") or "speaker0",
            )
            for rel in resolved["relations"]
        ],
        session_id="__calibration_normalize__",
        log_label="calibration",
    )
    before = merger.get_all_triples()
    # extraction_trace() re-entry is a no-op that yields the scope run_stage
    # already opened, which is where the pass's own nested scope lands the
    # ``normalize`` phase record.  The operator's prompt overrides are
    # already active for the whole run, opened once by the envelope.
    with extraction_trace() as trace:
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


def validate_anonymize_facts(state: dict, req: CalibrateAnonymizeFactsRequest) -> dict[str, Any]:
    """Resolve the fact list and prompt overrides for
    :func:`dispatch_anonymize_facts` — paired with it as the
    ``anonymize_facts`` stage's validate/dispatch split.
    """
    has_facts = req.facts is not None
    has_snapshot = req.snapshot_path is not None
    if has_facts == has_snapshot:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly one of 'facts' or 'snapshot_path' must be provided, "
                "not both and not neither."
            ),
        )
    resolved: dict[str, Any] = {"overrides": resolve_prompt_variants(state, req.prompt_variants)}
    facts = req.facts if has_facts else _relations_from_snapshot(req.snapshot_path)  # type: ignore[arg-type]
    if not facts:
        raise HTTPException(
            status_code=400,
            detail="No facts to anonymize (empty facts list, or snapshot has no edges).",
        )
    resolved["facts"] = facts
    return resolved


def dispatch_anonymize_facts(
    state: dict, req: CalibrateAnonymizeFactsRequest, resolved: dict[str, Any]
) -> tuple[Any, dict]:
    """Run the graph-tier anonymize step on the resolved facts — paired
    with :func:`validate_anonymize_facts`, which resolves *resolved*
    before this runs.

    ``anonymize()`` itself opens no phase-trace scope — ``paramem.cloud``
    must not import ``paramem.graph`` — and production's own call site
    (``paramem.training.graph_enrich.enrich_graph``) does not wrap it in
    one either, so this opens ``phase_trace("anonymize")`` itself, around
    the identical primitive call, the same way the session-tier
    ``anonymize`` stage body does.
    """
    from paramem.cloud.anonymize import anonymize
    from paramem.server.consolidation import get_or_create_consolidation_loop

    loop = get_or_create_consolidation_loop(state)
    ext_cfg = loop.extraction.config
    facts = resolved["facts"]
    # Mirrors enrich_graph's chunk_nodes: every distinct subject/object
    # surface across the facts this call anonymizes.
    identity_domain = sorted(
        {str(f.get("subject", "")) for f in facts if f.get("subject")}
        | {str(f.get("object", "")) for f in facts if f.get("object")}
    )
    with phase_trace("anonymize") as t:
        anon_prompt = _load_prompt("anonymization_facts.txt")
        anon_system = _load_prompt("anonymization_system.txt")
        payload = anonymize(
            facts,
            loop.model,
            loop.tokenizer,
            transcript="",
            scrub=ext_cfg.scrub,
            identity_domain=identity_domain,
            token_envelope=ext_cfg.anonymize_token_envelope,
            seed=req.params.seed,
            user_prompt_template=anon_prompt,
            system_prompt=anon_system,
        )
        t.set_raw(payload.raw)
        t.set_parsed(
            {
                "mapping": dict(payload.forward),
                "mapping_size": len(payload.forward),
                "status": payload.status,
                "failure": payload.failure,
                "slices": payload.slices,
                "slices_failed": payload.slices_failed,
            }
        )
    parsed: dict[str, Any] = {
        "status": payload.status,
        "failure": payload.failure,
        "mapping": dict(payload.forward),
        "slices": payload.slices,
        "slices_failed": payload.slices_failed,
        "identity_domain_size": len(identity_domain),
        "facts_count": len(facts),
    }
    return payload.raw, parsed


def validate_name(state: dict, req: CalibrateNameRequest) -> dict[str, Any]:
    """Resolve the prompt overrides for :func:`dispatch_name` — paired
    with it as the ``name`` stage's validate/dispatch split.
    """
    return {"overrides": resolve_prompt_variants(state, req.prompt_variants)}


def dispatch_name(
    state: dict, req: CalibrateNameRequest, resolved: dict[str, Any]
) -> tuple[Any, dict]:
    """Run the production name extractor on the request's turn list —
    paired with :func:`validate_name`, which resolves *resolved* before
    this runs.

    Calls :func:`~paramem.graph.name_extraction.extract_name_via_llm` — the
    same function, on the same base weights, that
    ``_run_enrollment_for_speaker`` calls in production, and which opens
    the ``name_extract`` phase itself.  Nothing here re-implements the
    post-filter or synthesises a phase record: ``prompts``, ``raw_output``
    and ``n_input_tokens`` all come from the phase the primitive opened.

    ``user_turns_only`` mirrors the production default (``True``) — only
    user turns reach the model; set to ``False`` to include assistant turns
    and reproduce the original (buggy) context-scoping behaviour for
    comparative testing.
    """
    from paramem.graph.name_extraction import extract_name_via_llm
    from paramem.models.loader import base_model_inference

    inference_params = {
        "temperature": req.params.temperature,
        "seed": req.params.seed,
        "max_tokens": req.params.max_tokens,
    }
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    with base_model_inference(model):
        extracted, raw_output = extract_name_via_llm(
            req.turns,
            model,
            tokenizer,
            user_turns_only=req.user_turns_only,
            params=inference_params,
        )
    return raw_output, {"name": extracted}


def validate_respond(state: dict, req: CalibrateRespondRequest) -> dict[str, Any]:
    """Validate one ``/calibrate/respond`` request — paired with
    :func:`dispatch_respond` as the ``respond`` stage's validate/dispatch
    split.

    Rejects empty ``text``, empty or unknown ``speaker_id``
    (``store.get_name(...) is None`` — **400, not 404**: the driver script
    at ``scripts/dev/calibrate_prompts.py`` turns any 404 into a
    ``calibrate_endpoint_enabled`` operator hint, which would mislead on an
    unenrolled speaker), a missing ``speaker_store``/``session_buffer``/
    ``router``/``memory_store`` (503 — server-not-ready, matching
    :func:`preflight`'s vocabulary), and an unresolvable prompt variant —
    all before any model call.
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty.")
    if not req.speaker_id:
        raise HTTPException(
            status_code=400,
            detail="speaker_id is required (no empty-string default).",
        )
    store = state.get("speaker_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Speaker store not ready.")
    if state.get("session_buffer") is None:
        raise HTTPException(status_code=503, detail="Session buffer not ready.")
    if state.get("router") is None:
        raise HTTPException(status_code=503, detail="Router not ready.")
    if state.get("memory_store") is None:
        raise HTTPException(status_code=503, detail="Memory store not ready.")
    if store.get_name(req.speaker_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker_id: {req.speaker_id!r} is not enrolled.",
        )
    return {"overrides": resolve_prompt_variants(state, req.prompt_variants)}


def dispatch_respond(
    state: dict, req: CalibrateRespondRequest, resolved: dict[str, Any]
) -> tuple[str, dict]:
    """Run one production serving turn for an enrolled speaker — paired
    with :func:`validate_respond`, which resolves *resolved* before this
    runs.

    A standalone use case, like :func:`dispatch_normalize`,
    :func:`dispatch_anonymize_facts`, and :func:`dispatch_name`: it never
    goes through :class:`~paramem.graph.extraction_pipeline.ExtractionPipeline`
    — calls :func:`~paramem.server.inference.handle_chat` verbatim, the same
    function ``POST /chat`` and ``POST /voice`` dispatch to, with the SAME
    production kwarg set. Everything the call reaches from there (dual-graph
    routing, the personal probe, HA escalation, cloud escalation,
    abstention, the base-model fallback) runs exactly as it would on a real
    turn: **this call may actuate a Home Assistant device and place a
    billed cloud call**, with no opt-out.

    Resolves the turn's language
    (:func:`~paramem.server.lang_id.resolve_text_language`), the speaker's
    display name, and the stored conversation history, then calls
    :func:`~paramem.server.inference.handle_chat`; ``model``/``tokenizer``
    are read straight from ``state``.

    Envelope semantics specific to this use case:

    * ``raw_output`` is the token-space reply (``speaker{N}`` tokens intact)
      — the exact string :meth:`~paramem.server.session_buffer.SessionBuffer.append`
      would persist for a real turn.  There is deliberately no
      ``resolved_text`` companion: resolving ``speaker{N}`` tokens to real
      household display names would write them into this on-disk artifact,
      contradicting the token-space storage discipline
      (:func:`~paramem.server.speaker.resolve_speaker_tokens` is scoped to
      human-display output only), and it would only duplicate
      ``raw_output`` in substance.
    * ``parsed`` carries ``escalated`` (bool), every key of
      :attr:`~paramem.server.inference.ChatResult.diagnostics` (routing
      decision, and — on the personal-probe leg — per-adapter probe counts
      and the temporal selection outcome) flattened in, and
      ``variants_unexercised`` (sorted list, empty when every override
      loaded): which prompt was resolved is branch-dependent (an
      HA-answered turn loads none of the serving prompts at all;
      ``cloud_serving_system.txt`` loads only on cloud escalation;
      ``recall_selection.txt`` only on the temporal personal leg;
      ``intent_classifier.txt`` only under ``intent.mode: llm``), so a
      variant that never got a chance to load is NOT an error — a
      multi-variant sweep must not fail because one leg didn't fire on
      this particular utterance, the turn still ran production-faithfully.
      Computed by diffing ``req.prompt_variants``'s keys against every
      ``<override:{name}>`` basename this dispatch's own trace captured.
    * ``params_effective`` is all-``null`` by construction: this request
      shape has no sampling-parameter field (see
      :class:`CalibrateRespondRequest`), so the route passes a bare
      :class:`CalibrateParams` with ``supports_seed=False`` — no seed
      threads into the serving path at all.
    * ``n_input_tokens`` measures the template of the FIRST non-``*_system.txt``
      prompt the ``serve_turn`` phase record captured (see
      :func:`_provenance_from_records`) — which basename that actually is
      depends on which branch the turn took: ``intent_classifier.txt``
      under the shipped ``intent.mode: llm``, ``recall_selection.txt`` on
      the temporal personal leg under encoder mode, or
      ``serving_directives.txt`` otherwise.  Always the unsubstituted
      template, per the project-wide provenance contract, never the
      rendered prompt actually sent.
    * ``phases`` includes any nested cloud-egress phases (``local_extract``,
      ``cloud_enrich``, …) that a turn escalating through
      :func:`~paramem.server.inference.answer_via_cloud` opened on this same
      trace.

    This call writes no session-buffer entry, no speaker-store write, and no
    registry write (:func:`~paramem.server.inference.handle_chat` itself
    performs none); the reply and routing diagnostics are written to this
    run's own artifact directory by the shared
    :func:`~paramem.utils.artifacts.on_calibration_result` hook, same as
    every other calibration stage.
    """
    from paramem.server.inference import handle_chat

    model, tokenizer = state["model"], state["tokenizer"]
    language, _ = lang_id.resolve_text_language(req.text, state["config"].text_lang_detection)
    store = state["speaker_store"]
    speaker_name = store.resolve_speaker_name(req.speaker_id)
    history = state["session_buffer"].get_conversation_turns(req.conversation_id)
    with extraction_trace() as trace:
        result = handle_chat(
            text=req.text,
            conversation_id=req.conversation_id,
            speaker=speaker_name,
            speaker_id=req.speaker_id,
            history=history,
            model=model,
            tokenizer=tokenizer,
            config=state["config"],
            router=state["router"],
            cloud_agent=state.get("cloud_agent"),
            ha_client=state.get("ha_client"),
            language=language,
            effective_mode=state.get("effective_mode"),
            memory_store=state["memory_store"],
        )
    prompts, _ = _provenance_from_records(trace.records, "serve_turn")
    exercised = {
        path[len("<override:") : -1]
        for p in prompts
        if isinstance((path := p.get("path")), str) and path.startswith("<override:")
    }
    parsed = {
        "escalated": result.escalated,
        **result.diagnostics,
        "variants_unexercised": sorted(set(req.prompt_variants) - exercised),
    }
    return result.text, parsed


def validate_extract_pending(state: dict, req: CalibrateExtractPendingRequest) -> dict[str, Any]:
    """Resolve the prompt overrides for ``POST /calibrate/extract_pending``.

    No session-selection field exists on the request (see
    :class:`CalibrateExtractPendingRequest`): the run's promise is "exactly
    what a fold would extract right now", so there is nothing else to
    validate here.  The dispatch closure itself is built at the route
    (``paramem.server.app``), since its step —
    :func:`~paramem.server.app._extract_pending_sessions` — is a
    server-layer function, not one this module can reach without
    reaching past the single extraction-graph lifetime owner.
    """
    return {"overrides": resolve_prompt_variants(state, req.prompt_variants)}


def _effective_params(params: CalibrateParams, *, supports_seed: bool) -> dict:
    """Return the params dict the call effectively applied.

    ``supports_seed`` distinguishes local stages (where seed is honoured
    via a scoped torch.Generator) from a stage fronting an API that accepts
    no seed parameter (where the field would be silently dropped).  The
    response uses this to inform the operator which fields actually
    landed.

    Every local stage's route passes ``supports_seed=True`` on its
    :class:`CalibrationRunSpec`. ``/calibrate/respond`` is the one
    ``supports_seed=False`` caller — its request shape carries no sampling
    parameters at all, so ``seed`` is forced to ``null`` rather than echoing
    a value that was never collected.  top_p / top_k are not yet threaded to
    any stage's generation call (documented gap).  The field is reported
    as-requested for transparency.
    """
    out: dict = {}
    for f in ("temperature", "top_p", "top_k", "max_tokens"):
        out[f] = getattr(params, f)
    out["seed"] = params.seed if supports_seed else None
    return out


# ---------------------------------------------------------------------------
# One declaration, one spec builder, one executor — consumed by every
# /calibrate/* route AND by every test that drives a run without going
# through app.py's FastAPI handlers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StandaloneDeclaration:
    """One standalone (non-chain) calibration stage's fixed shape — the
    same declared-once role :data:`_CHAIN` plays for the five chain
    routes, for the remaining four (``normalize``, ``anonymize_facts``,
    ``name``, ``respond``).  Read by both the production route dispatcher
    (:func:`build_spec`, called from ``paramem.server.app``) and every
    test caller building a run through :func:`build_spec` directly, so a
    stage's route path, input-prompt phase, seed support, and params
    source are declared exactly once rather than once per route handler
    and once per test.

    Attributes:
        route_path: The producing route (``"/calibrate/normalize"``).
        input_prompt_phase: Which phase record's non-system prompt feeds
            ``n_input_tokens`` and the provenance block.
        supports_seed: Whether ``params_effective["seed"]`` echoes the
            request's seed.
        validate: The stage's ``validate_*`` function — ``(state, req) ->
            resolved dict``.
        dispatch: The stage's ``dispatch_*`` function — ``(state, req,
            resolved) -> (raw_output, parsed)``.
        params: Derives the spec's :class:`CalibrateParams` from the
            request — ``req.params`` for every stage except ``respond``,
            whose request shape carries no sampling-parameter field at
            all.
    """

    route_path: str
    input_prompt_phase: str
    supports_seed: bool
    validate: "Callable[[dict, Any], dict[str, Any]]"
    dispatch: "Callable[[dict, Any, dict[str, Any]], tuple[Any, Any]]"
    params: "Callable[[Any], CalibrateParams]"


_STANDALONE: dict[str, _StandaloneDeclaration] = {
    "normalize": _StandaloneDeclaration(
        route_path="/calibrate/normalize",
        input_prompt_phase="normalize",
        supports_seed=True,
        validate=validate_normalize,
        dispatch=dispatch_normalize,
        params=lambda req: req.params,
    ),
    "anonymize_facts": _StandaloneDeclaration(
        route_path="/calibrate/anonymize_facts",
        input_prompt_phase="anonymize",
        supports_seed=True,
        validate=validate_anonymize_facts,
        dispatch=dispatch_anonymize_facts,
        params=lambda req: req.params,
    ),
    "name": _StandaloneDeclaration(
        route_path="/calibrate/name",
        input_prompt_phase="name_extract",
        supports_seed=True,
        validate=validate_name,
        dispatch=dispatch_name,
        params=lambda req: req.params,
    ),
    "respond": _StandaloneDeclaration(
        route_path="/calibrate/respond",
        input_prompt_phase="serve_turn",
        # CalibrateRespondRequest carries no sampling-parameter field at
        # all — no seed threads into the serving path.
        supports_seed=False,
        validate=validate_respond,
        dispatch=dispatch_respond,
        params=lambda _req: CalibrateParams(),
    ),
}


def route_path_for(stage: str) -> str:
    """The route path for a declared calibration stage.

    :data:`_CHAIN` and :data:`_STANDALONE` combined cover every
    ``/calibrate/*`` route except ``extract_pending``, whose spec is built
    at its own route (see :func:`validate_extract_pending`'s docstring).
    """
    if stage in _CHAIN:
        return f"/calibrate/{stage}"
    return _STANDALONE[stage].route_path


def build_spec(
    stage: str,
    state: dict,
    req: Any,
    *,
    run_id: str,
    artifact_dir: Path,
) -> CalibrationRunSpec:
    """Build one calibration run's :class:`CalibrationRunSpec` — the single
    declaration every route and every test consumes, so a stage's shape
    (route path, input-prompt phase, seed support, params source) is
    declared exactly once.

    Runs the stage's ``validate_*`` as a side effect (raises ``HTTPException``
    on a bad request, exactly as a live dispatch would, before any model
    call — this is meant to run on the event loop, ahead of dispatch).

    Args:
        stage: A chain use case (a key of :data:`_CHAIN`) or a standalone
            stage (a key of :data:`_STANDALONE`) — every ``/calibrate/*``
            stage except ``extract_pending``, which builds its own spec at
            the route.
        state: The live server state dict, passed to ``validate_*`` and
            closed over by the built ``dispatch`` callable.
        req: The stage's own validated request model.
        run_id: This run's stamp, minted by the caller.
        artifact_dir: This run's directory, minted by the caller.

    Returns:
        A :class:`CalibrationRunSpec` ready for :func:`run_stage`.
    """
    if stage in _CHAIN:
        resolved = validate_chain(state, stage, req)
        return CalibrationRunSpec(
            stage=stage,
            route_path=f"/calibrate/{stage}",
            run_id=run_id,
            artifact_dir=artifact_dir,
            dispatch=lambda: dispatch_chain(state, stage, req, resolved),
            input_prompt_phase=resolved["focus"],
            supports_seed=True,
            params=req.params,
            overrides=resolved["overrides"],
            evicts_voice=resolved["evicts_voice"],
        )
    decl = _STANDALONE[stage]
    resolved = decl.validate(state, req)
    return CalibrationRunSpec(
        stage=stage,
        route_path=decl.route_path,
        run_id=run_id,
        artifact_dir=artifact_dir,
        dispatch=lambda: decl.dispatch(state, req, resolved),
        input_prompt_phase=decl.input_prompt_phase,
        supports_seed=decl.supports_seed,
        params=decl.params(req),
        overrides=resolved["overrides"],
    )
