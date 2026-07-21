"""LLM-based knowledge graph extraction — generate once, parse once."""

import contextlib
import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from paramem.evaluation.recall import generate_answer
from paramem.graph.cloud_egress import (
    _DEFAULT_ANONYMIZER_MAX_TOKENS,
    _DEFAULT_FILTER_MAX_TOKENS,
    _DEFAULT_FILTER_TEMPERATURE,
    _DEFAULT_FILTER_TIMEOUT_SECONDS,
    AnonymizedPayload,
    CloudScope,
    DeanonResult,
    _extract_json_block,
    anonymize_for_cloud,
    deanonymize_facts,
    deanonymize_response_text,
)
from paramem.graph.entity_correction import correct_entity_surfaces
from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.phase_trace import chain_stopped, extraction_trace, phase_trace
from paramem.graph.placeholders import (
    _FACT_FIELDS,
    _normalize_anonymization_mapping,
    _substitute_whole_words,
    braced,
)
from paramem.graph.prompts import _load_prompt
from paramem.graph.relation_build import (
    CAUSE_ANON_JUDGE,
    CAUSE_CLOUD_EMPTY,
    CAUSE_DEANON_JUDGE,
    CAUSE_DEANON_SUBSTITUTION,
    CAUSE_SCALAR_PARTITION,
    CAUSE_SCHEMA_VALIDATION,
    apply_rebuild,
    build_relations,
    partition_scalar_facts,
    recovery_gate,
)
from paramem.graph.schema import SessionGraph
from paramem.graph.schema_config import (
    fallback_entity_type,
    fallback_relation_type,
    relation_types,
)
from paramem.models.loader import adapt_messages, base_model_inference
from paramem.server.session_buffer import SessionBuffer
from paramem.server.vram_guard import vram_scope
from paramem.utils.cloud_admission import (
    OPENAI_COMPAT_ENDPOINTS,
    OPENAI_COMPAT_PROVIDERS,
    PROVIDER_KEY_ENV,
    evaluate_cloud_egress,
)
from paramem.utils.identity import canonical, is_speaker_id

logger = logging.getLogger(__name__)


class ExtractionFailed(RuntimeError):
    """Raised when a load-bearing extraction phase fails and the cycle
    must be aborted for this session.

    Currently raised from the ``sota_enrich`` phase when the cloud
    enrichment call fails (parse failure or upstream non-2xx — including
    Anthropic 529 overloaded), because falling back to pre-enrichment
    facts silently bakes a degraded snapshot into the cumulative graph.

    The per-session caller (``_extract_and_start_training`` /
    ``_extract_and_start_training`` in ``app.py``) catches this and
    treats it like ``VramExhausted``: log, leave the session pending
    (skip ``mark_consolidated``), continue with the next session.  The
    cumulative graph is unmodified because the failure propagates
    BEFORE :meth:`ConsolidationLoop.extract_session` reaches the merge
    call.

    ``phase`` names the extraction phase that failed (e.g.
    ``"sota_enrich"``).  ``reason`` is a short operator-facing string.
    """

    def __init__(self, phase: str, reason: str) -> None:
        super().__init__(f"{phase}: {reason}")
        self.phase = phase
        self.reason = reason


# Single output-token budget for every LLM call in the extraction pipeline:
# local extraction, anonymization, SOTA enrichment, plausibility (local +
# cloud), graph-level enrichment. Threaded through extract_graph →
# _sota_pipeline → all sub-functions so a single
# ``ConsolidationLoop.extraction_max_tokens`` (server.yaml
# ``consolidation.extraction_max_tokens``) governs the whole chain.
#
# 8192 is sized for Mistral 7B against ~1500-word document chunks (the local
# chunker's max). Empirical worst-case observed output for a dense resume
# chunk was ~2200 tokens; 8192 gives ~3.7× headroom. If the chunker contract
# changes, revisit jointly with that change.
#
# Plausibility output couples to chunk density. The filter's contract
# (configs/prompts/sota_plausibility.txt) is "Return ONLY a JSON array of
# surviving facts, schema unchanged" — so its output volume scales with
# the surviving-fact count, which scales with chunk density. Lowering the
# cap independently for plausibility was attempted and reverted: a 2048
# cap truncated the JSON array on dense chunks, the parse failed, and the
# caller fell back to passing the unfiltered set forward. KV-cache
# pressure must be mitigated upstream (STT/TTS eviction, gc.collect
# before empty_cache, per-phase vram_scope wraps), not by truncating
# correctness-bearing output.
# _DEFAULT_FILTER_MAX_TOKENS / _DEFAULT_FILTER_TEMPERATURE /
# _DEFAULT_FILTER_TIMEOUT_SECONDS now live in paramem.graph.cloud_egress
# (imported above) — the anonymizer's own default and this module's
# SOTA-call defaults share the one source of truth; see that module's
# docstring for why the constant lives there rather than here (avoiding
# the extractor<->cloud_egress import cycle).


# ---------------------------------------------------------------------------
# WSL2 GPU wake helper — covers the post-cloud-call → next-GPU-op gap.
#
# Background: WSL2 + RTX 5070 + Modern Standby lets the GPU enter a low-power
# state after ~60s of idle. A SOTA cloud round-trip is a typical trigger
# (anonymization completes → cloud SOTA call takes 30–90s → next local-LLM
# call hits "device not ready" before the driver is fully back). Once that
# first op fails, PyTorch's allocator bookkeeping is corrupted with
# ``INTERNAL ASSERT FAILED in CUDACachingAllocator`` and no retry can
# recover — only a server restart will. The strategy is therefore to PREVENT
# the first attempt from failing via a wall-clock settle on detection.
#
# The same pattern (different trigger — post-training-pass instead of
# post-cloud-idle) is documented in ``paramem/server/gates.py``
# ``_settle_cuda_and_load_adapter``. Constants here are aligned with that
# helper but tuned for the cloud-idle path.
# ---------------------------------------------------------------------------

# Markers indicating PyTorch's CUDA allocator is corrupted. Mirrored from
# ``gates.py:_CUDA_TERMINAL_MARKERS``.
_CUDA_TERMINAL_MARKERS: tuple[str, ...] = ("INTERNAL ASSERT FAILED", "CUDACachingAllocator")
# Up to 3 attempts (1 + 2 retries). Beyond that, server restart is needed.
_GPU_WAKE_RETRY_COUNT: int = 3
# 5s wall-clock settle per retry. Empirically a 60s idle gap surfaced
# "device not ready" once; 5s × 2 retries (10s total) covers the WSL2
# driver wake-up latency observed on this host.
_GPU_WAKE_SETTLE_SECONDS: float = 5.0


def _vram_snapshot(label: str) -> None:
    """Log GPU memory state for telemetry around major pipeline calls.

    Used to localise VRAM-pressure-induced crashes in the SOTA pipeline.
    Output is grep-friendly:
    ``VRAM <label>: alloc=NNNN MiB reserved=NNNN MiB peak=NNNN MiB
                   smi_used=NNNN MiB smi_free=NNNN MiB``.

    The ``smi_*`` fields query ``nvidia-smi`` so the gap between
    PyTorch's accounted-for memory and the host-visible total surfaces
    dxg/host-side allocations that the WSL2 paravirt layer holds outside
    of PyTorch's view — that gap is what crashed us under
    ``dxgkio_make_resident`` ENOMEM.

    Resets peak after reading so each window's contribution is visible
    on the next snapshot. No-op when CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        return
    try:
        if not torch.cuda.is_available():
            return
    except Exception:  # noqa: BLE001 — MagicMock test stubs may raise
        return
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mib = torch.cuda.memory_reserved() / (1024 * 1024)
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001
        logger.debug("VRAM snapshot %s: query failed: %s", label, exc)
        return
    smi_used = smi_free = -1.0
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
            smi_used = float(parts[0].strip())
            smi_free = float(parts[1].strip())
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "VRAM %s: alloc=%.0f MiB reserved=%.0f MiB peak=%.0f MiB "
        "smi_used=%.0f MiB smi_free=%.0f MiB",
        label,
        alloc_mib,
        reserved_mib,
        peak_mib,
        smi_used,
        smi_free,
    )


def _summarise_graph(graph: SessionGraph) -> dict:
    """Compact, JSON-serialisable view of a SessionGraph for phase traces.

    The full graph is large and largely redundant across phases; the
    summary captures what changes between phases — entity names + types
    + speaker_id markers, and relation triples.  Calibration consumers
    diff these dicts to see exactly what each phase produced or mutated.
    """
    return {
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
        "entity_names": [e.name for e in graph.entities],
        "entity_types": {e.name: e.entity_type for e in graph.entities},
        "speaker_entities": [
            {
                "name": e.name,
                "entity_type": e.entity_type,
                "speaker_id": e.speaker_id,
                "attributes": dict(e.attributes) if e.attributes else {},
            }
            for e in graph.entities
            if e.speaker_id
        ],
        "triples": [[r.subject, r.predicate, r.object] for r in graph.relations],
    }


def _wait_for_gpu_ready(*, pre_settle_seconds: float = 10.0) -> None:
    """Settle the GPU before a CUDA op that follows a long idle gap.

    The WSL2 driver needs wall-clock time after a long idle to be safely
    callable again — ``torch.cuda.synchronize`` returns too quickly to
    cover the gap (documented in ``gates.py:_settle_cuda_and_load_adapter``,
    same root behaviour, different trigger). A trivial ``torch.zeros``
    probe alone is also insufficient: it succeeds on a sleepy driver, but
    the next real ``model.generate`` still crashes. We therefore sleep
    unconditionally for ``pre_settle_seconds`` first, then probe to catch
    the residual cases where the driver still isn't ready.

    On "device not ready" from the probe: additional wall-clock retries
    (up to ``_GPU_WAKE_RETRY_COUNT`` total attempts × ``_GPU_WAKE_SETTLE_SECONDS``).
    On allocator-corruption markers: bail immediately — retries cannot
    recover.

    No-op when CUDA is unavailable (CPU-only test environments). The
    settle is also skipped when ``pre_settle_seconds <= 0``.

    Default ``pre_settle_seconds=10.0`` matches
    ``gates.py:_MOUNT_INITIAL_SETTLE_SECONDS`` — empirically required after
    a heavy GPU pass + cloud-idle gap.

    Raises ``RuntimeError`` if the GPU is still not ready after retries.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    # Pre-emptive settle. Cheap (10s wall-clock) compared to a corrupted-
    # allocator restart cycle (~30s server boot + lost cycle).
    if pre_settle_seconds > 0:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("pre-probe CUDA settle failed: %s", exc)
        time.sleep(pre_settle_seconds)

    last_exc: BaseException | None = None
    for attempt in range(_GPU_WAKE_RETRY_COUNT):
        try:
            torch.cuda.synchronize()
            torch.zeros(1, device="cuda")
            if attempt > 0:
                logger.info(
                    "GPU wake recovered on attempt %d/%d",
                    attempt + 1,
                    _GPU_WAKE_RETRY_COUNT,
                )
            return
        except RuntimeError as exc:
            msg = str(exc)
            if any(m in msg for m in _CUDA_TERMINAL_MARKERS):
                logger.error(
                    "GPU wake: CUDA allocator corruption detected — "
                    "server restart required to recover: %s",
                    msg,
                )
                raise
            if "device not ready" not in msg.lower():
                raise
            last_exc = exc
            if attempt < _GPU_WAKE_RETRY_COUNT - 1:
                logger.warning(
                    "GPU wake attempt %d/%d: 'device not ready' — settling %ss",
                    attempt + 1,
                    _GPU_WAKE_RETRY_COUNT,
                    _GPU_WAKE_SETTLE_SECONDS,
                )
                time.sleep(_GPU_WAKE_SETTLE_SECONDS)
    assert last_exc is not None
    raise last_exc


# Prompt filename constants — one definition site for the single
# extraction-prompt source of truth.  The transcript prompt-pair is
# used for every source_type; document chunks land in the same
# ``{transcript}`` slot at the chat-template layer.  Per-source
# extension goes via overrides or by prepending/appending content to
# the slot at the caller layer — never via parallel file pairs.  The
# old DOCUMENT_*_FILENAME constants and their backing files are
# retired (would silently drift on schema-shape rules).
DEFAULT_SYSTEM_PROMPT_FILENAME = "extraction_system.txt"
DEFAULT_USER_PROMPT_FILENAME = "extraction.txt"
DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME = "extraction_procedural.txt"


def build_speaker_context(
    speaker_id: str | None,
    speaker_name: str | None,
) -> str:
    """Single source of truth for the extraction-prompt speaker directive.

    Loads the ``EXTRACTION-DIRECTIVE`` section from
    ``configs/prompts/speaker_directive.txt`` and slots in
    ``{speaker_id}`` (the stable system id, e.g. ``"speaker0"``) and
    ``{speaker_name}`` (the display name, e.g. ``"Alice"``).

    Returns an empty string when ``speaker_id`` is absent or empty —
    leaving the ``{speaker_context}`` slot in the few-shots blank (the
    prompt's note tells the model never to emit the ``{{SPEAKER_NAME}}``
    literal).

    The display name is injected as COMPREHENSION CONTEXT so the model
    can map self-references ("I", "my name is Alice", etc.) onto the
    stable lowercase ``speaker{N}`` id while keeping any same-named third
    party separate.  The subject of every extracted speaker-fact must be
    ``speaker_id``, not the display name.

    Args:
        speaker_id: System speaker id (e.g. ``"speaker0"``).  When
            empty or ``None``, the directive is suppressed entirely.
        speaker_name: Display name resolved from the speaker store (e.g.
            ``"Alice"``).  ``None`` when unknown or anonymous; the
            directive uses the id string in place of the display name so
            the model still binds onto the id.
    """
    if not speaker_id:
        return ""
    from paramem.graph.prompts import _load_speaker_directive_section

    template = _load_speaker_directive_section("EXTRACTION-DIRECTIVE")
    # When no display name is known, use the id itself as comprehension
    # context so the placeholder reference is consistent throughout.
    effective_name = speaker_name or speaker_id
    return "\n" + template.format(speaker_id=speaker_id, speaker_name=effective_name) + "\n"


def load_extraction_prompts(
    prompts_dir: str | Path | None = None,
    *,
    system_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_filename: str = DEFAULT_USER_PROMPT_FILENAME,
    model: str | None = None,
) -> tuple[str, str]:
    """Load extraction prompts from a directory, with hardcoded fallbacks.

    The prompts this function loads are external config — edit the files
    under ``configs/prompts/`` to tune extraction behaviour; no code
    changes are needed.

    Args:
        prompts_dir: Directory containing the prompt files.  Falls back to
                     ``configs/prompts/`` in the project root, then to
                     hardcoded defaults.
        system_filename: Filename of the system prompt.  Defaults to
                         :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
                         (``"extraction_system.txt"``).  Used for every
                         source type — there is no separate document
                         variant; document chunks land in the
                         ``{transcript}`` slot of the same prompt.
        user_filename: Filename of the user-turn prompt template.
                       Defaults to :data:`DEFAULT_USER_PROMPT_FILENAME`
                       (``"extraction.txt"``).  Used for every source
                       type for the same reason.
        model: Model alias (e.g. ``"qwen3-4b"``).  When provided,
               resolution is per-file: ``prompts_dir/<model>/<filename>``
               is tried first, falling back to ``prompts_dir/<filename>``
               and then ``configs/prompts/<filename>``.  Only local-model
               extraction prompts are per-model; SOTA cloud prompts call
               with ``model=None`` (unchanged).

    Returns:
        ``(system_prompt, extraction_prompt)`` tuple.
    """
    pd = Path(prompts_dir) if prompts_dir else None
    system = _load_prompt(system_filename, prompts_dir=pd, model=model, required=True)
    prompt = _load_prompt(user_filename, prompts_dir=pd, model=model, required=True)
    return system, prompt


def extract_procedural_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    speaker_id: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    speaker_name: str | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
    model_alias: str | None = None,
    timestamp: str | None = None,
    source_type: str = "transcript",
) -> SessionGraph:
    """Extract preferences/habits from a session transcript.

    Separate extraction pass with a dedicated prompt targeting
    behavioral patterns rather than factual knowledge. Runs through
    :func:`_run_local_extraction` — the SAME shared generate→parse→trace
    primitive :func:`extract_graph` uses for its ``local_extract`` and
    ``second_order_extract`` phases — under its own
    :func:`~paramem.graph.phase_trace.extraction_trace` scope, self-tracing
    the ``procedural_extract`` phase. Callers no longer need to open a
    phase trace around this call; nesting inside an outer
    ``extraction_trace`` (e.g. the session-level trace opened by
    consolidation) is a no-op, so the phase record lands wherever the
    call happens to be nested.

    Because parsing now goes through the shared primitive, this pass gets
    the same improvements ``local_extract``/``second_order_extract``
    already have: ``raw_output`` and prompt provenance on the phase
    record, ``outcome="failed"`` recorded on parse error, and tolerance
    for a bare-list model output shape.

    Args:
        timestamp: Session-start assertion time (ISO 8601), typically the
            session's ``started_at``. Stamped onto the returned
            ``SessionGraph.timestamp`` so ``last_seen`` on newly-merged
            edges reflects when the facts were asserted, not when
            extraction ran. ``None`` (default) falls back to ``now()`` —
            preserves behaviour for callers that don't yet have a real
            session-start time.
        speaker_name: Display name of the speaker (e.g. from voice enrollment).
            Passed to ``build_speaker_context`` as comprehension context so the
            model can map self-references in the transcript onto the stable
            ``speaker_id``.  The model uses the ``speaker{N}`` id — not the
            display name — as the subject of every extracted preference.
            Mirrors the same parameter on ``extract_graph``.
        speaker_id: Speaker store ID (e.g. ``"speaker0"``). Stamped onto every
            ``Relation`` extracted in this pass as provenance. Required —
            callers must always supply a real speaker ID.
        seed: Optional RNG seed forwarded to :func:`generate_answer`.  At the
            default ``temperature=0.0`` (greedy decoding) this is a strict
            no-op.  Default ``None`` preserves production behaviour unchanged.
        system_prompt_filename: Filename of the system prompt within the prompts
            directory.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`.
            One prompt-pair is the single ground truth for procedural
            extraction; document chunks land in the ``{transcript}`` slot
            of the same prompt.
        user_prompt_filename: Filename of the user-turn prompt template.
            Defaults to :data:`DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME`
            (``"extraction_procedural.txt"``).  Same prompt for every
            source type.
        model_alias: Model alias (e.g. ``"qwen3-4b"``).  Enables per-file
            prompt resolution: ``prompts_dir/<model_alias>/<filename>``
            is checked first, falling back to ``prompts_dir`` and then
            ``configs/prompts/``.  SOTA cloud prompts are not affected.
        source_type: ``"transcript"`` (default) or ``"document"``. Forwarded
            to :func:`_stamp_speaker_entity` as the Guard B gate for the
            document-only exact-full-name rewrite of third-person speaker
            mentions onto ``speaker_id``.
    """
    with extraction_trace() as trace:
        graph = _run_local_extraction(
            model,
            tokenizer,
            transcript,
            session_id,
            speaker_id,
            temperature,
            max_tokens,
            prompts_dir,
            speaker_name,
            system_prompt_filename,
            user_prompt_filename,
            model_alias,
            seed,
            timestamp,
            source_type,
            phase_name="procedural_extract",
            vram_label="procedural",
        )
        trace.attach_to(graph)
    return graph


# Filename of the second-order extraction pass — it extracts facts ABOUT
# the named entities local_extract (first-order) surfaced, recovering a
# named relative's own attribute (location, job, trait) when Mistral 7B
# collapses a single-relative clause ("my brother Nadeem lives in Porto")
# into ONE relation instead of two. Reuses DEFAULT_SYSTEM_PROMPT_FILENAME —
# no second-order-specific system prompt.
DEFAULT_SECOND_ORDER_USER_PROMPT_FILENAME = "extraction_second_order.txt"


def _run_local_extraction(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    speaker_id: str,
    temperature: float,
    max_tokens: int,
    prompts_dir: str | Path | None,
    speaker_name: str | None,
    system_prompt_filename: str,
    user_prompt_filename: str,
    model_alias: str | None,
    seed: int | None,
    timestamp: str | None,
    source_type: str,
    *,
    phase_name: str,
    vram_label: str = "extract_main",
) -> SessionGraph:
    """Shared generate→parse→summarise→trace primitive for a local-model
    extraction phase.

    This is the ONE shared implementation for EVERY local-model extraction
    pass — ``extract_graph``'s ``local_extract`` (first-order: extracts
    from the raw transcript) and ``second_order_extract`` (second-order:
    re-extracts facts about the named non-speaker people ``local_extract``
    surfaced), plus ``extract_procedural_graph``'s ``procedural_extract``
    (behavioral-pattern pass) — differing only by which prompt is loaded
    (``user_prompt_filename``), the ``phase_name`` recorded on the trace,
    and the ``vram_label`` used for the generate's VRAM scope. Must be
    called from inside an active
    :func:`~paramem.graph.phase_trace.extraction_trace` scope (all call
    sites are — via :func:`extract_graph` or :func:`extract_procedural_graph`).

    On parse failure, records ``outcome="failed"`` on the phase trace and
    returns an empty :class:`SessionGraph` for ``session_id``/``timestamp``
    — mirrors pre-carve-out ``local_extract`` behaviour. The caller decides
    what an empty result means (``local_extract``'s caller returns
    immediately; ``second_order_extract``'s caller has nothing to union).
    """
    with phase_trace(phase_name) as t:
        raw_output = _generate_extraction(
            model,
            tokenizer,
            transcript,
            temperature,
            max_tokens,
            prompts_dir,
            speaker_name,
            speaker_id=speaker_id,
            system_prompt_filename=system_prompt_filename,
            user_prompt_filename=user_prompt_filename,
            model_alias=model_alias,
            seed=seed,
            vram_label=vram_label,
        )
        t.set_raw(raw_output)
        logger.debug("Raw extraction output (%s): %s", phase_name, raw_output[:500])
        try:
            graph = _parse_extraction(
                raw_output,
                session_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                timestamp=timestamp,
                source_type=source_type,
            )
        except Exception as exc:
            logger.warning(
                "%s parsing failed (%s), returning empty graph",
                phase_name,
                exc,
            )
            t.set_outcome("failed", reason=f"{type(exc).__name__}: {exc}")
            t.set_parsed({"entity_count": 0, "relation_count": 0})
            return SessionGraph(
                session_id=session_id,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            )
        t.set_parsed(_summarise_graph(graph))
    return graph


def _has_named_non_speaker_person(graph: SessionGraph) -> bool:
    """Gate for the ``second_order_extract`` phase: does the pass-1 graph
    contain a named (proper-name) person entity other than a speaker id?

    ``local_extract`` is speaker-centric, so a clause naming a non-speaker
    person by relationship ("my brother Nadeem lives in Porto") tends to
    keep only the speaker->person edge and drop that person's OWN fact
    (measured on Mistral 7B). A surviving named non-speaker person entity
    is exactly the set of people this failure mode can hit, and exactly
    what ``second_order_extract`` re-extracts facts about — so it is the
    gate: no such entity means nothing to recover, and the caller skips
    the phase entirely (no LLM call, no phase_trace record).
    """
    return any(e.entity_type == "person" and not is_speaker_id(e.name) for e in graph.entities)


# ---------------------------------------------------------------------------
# SESSION_EXTRACT — the declarative flow ``extract_graph`` walks via
# ``paramem.graph.flow.run_flow``.
#
# This is flow TOPOLOGY co-located with the primitives it calls (rather than
# a dedicated ``flows.py``) to avoid a circular import: the stage bodies below
# call ``_run_local_extraction`` / ``_sota_pipeline``, both defined in
# this module. Move this to a dedicated
# flow module once a second flow (beyond this one and the still-imperative
# ``extract_procedural_graph``) exists to justify the split.
#
# Each stage body is a VERBATIM lift of the corresponding block that used to
# live inline in ``extract_graph`` (``local_extract``, ``second_order_extract``,
# ``sota_pipeline``) or in the tail of ``_sota_pipeline`` (``deanonymize``,
# ``rebuild``) — same primitives, same arguments, same ``phase_trace`` scopes.
# ``run_flow`` does not open phases itself; every ``phase_trace`` call below is
# exactly the one the imperative version made.
# ---------------------------------------------------------------------------


def _stage_local_extract(ctx: StageContext, state: StageState) -> StageState:
    """``local_extract`` stage body — always runs.

    Local-model extraction step. Raw output is the canonical isolation
    point for the extraction prompt (calibration/debugging diffs prompt
    variants by comparing this raw_output, before any downstream phase
    has had a chance to mutate the result).
    """
    graph = _run_local_extraction(
        ctx.model,
        ctx.tokenizer,
        ctx.transcript,
        ctx.session_id,
        ctx.speaker_id,
        ctx.temperature,
        ctx.max_tokens,
        ctx.prompts_dir,
        ctx.speaker_name,
        ctx.system_prompt_filename,
        ctx.user_prompt_filename,
        ctx.model_alias,
        ctx.seed,
        ctx.timestamp,
        ctx.source_type,
        phase_name="local_extract",
    )
    return StageState(graph=graph)


def _stage_second_order_extract(ctx: StageContext, state: StageState) -> StageState:
    """``second_order_extract`` stage body — gated by
    :func:`_has_named_non_speaker_person` (the flow's ``applies_when``).

    Extracts facts ABOUT the named entities ``local_extract`` (first-order)
    surfaced, recovering a named relative's own attribute (location, job,
    trait) when local_extract collapsed a single-relative clause ("my
    brother Nadeem lives in Porto") into ONE relation instead of two
    (measured Mistral 7B failure mode).
    """
    graph = state.graph
    second_order_graph = _run_local_extraction(
        ctx.model,
        ctx.tokenizer,
        ctx.transcript,
        ctx.session_id,
        ctx.speaker_id,
        ctx.temperature,
        ctx.max_tokens,
        ctx.prompts_dir,
        ctx.speaker_name,
        ctx.system_prompt_filename,
        DEFAULT_SECOND_ORDER_USER_PROMPT_FILENAME,
        ctx.model_alias,
        ctx.seed,
        ctx.timestamp,
        ctx.source_type,
        phase_name="second_order_extract",
    )
    # Plain union: the second-order pass contributes recovered facts
    # (recall) — it is not a dedup boundary. Predicate-surface drift for a
    # fact both passes capture (e.g. "picked_up" vs "picks_up") is a
    # PRE-EXISTING pipeline phenomenon (the same drift already happens
    # across sessions) and is handled downstream exactly as cross-session
    # drift is: triple-identity dedup at GraphMerger._upsert_relation Case 1
    # (paramem/graph/merger.py:580) and, when enabled,
    # refinement_normalization's (subject, object)-grouped predicate-synonym
    # fold. A redundant near-dup key is benign — not a wrong answer, at
    # worst a redundant indexed key — so it is deliberately NOT special-cased
    # here; filtering on (subject, object) identity would also destroy
    # genuinely distinct same-(s,o) facts (e.g. born_in + lives_in).
    graph.relations.extend(second_order_graph.relations)
    graph.entities.extend(second_order_graph.entities)
    return StageState(graph=graph)


def _session_egress_permitted(ctx: StageContext) -> bool:
    """``sota_pipeline``'s cloud-admission gate, as a flow predicate.

    Routes the session-tier question through the one shared component
    (:func:`~paramem.utils.cloud_admission.evaluate_cloud_egress`) rather
    than restating its terms as a boolean expression in the stage spec.
    Logs every unmet term in a single line when the answer is no — a flow
    ``enabled_when`` skip is otherwise silent, and the operator whose
    ``sota_enabled`` is on but whose key is unset needs to be told which
    term failed.

    Args:
        ctx: The run-constant flow context.

    Returns:
        ``True`` when a cloud enrichment call may be placed for this run.
    """
    verdict = evaluate_cloud_egress(
        sota_enabled=ctx.sota_enabled,
        provider=ctx.noise_filter,
        model=ctx.noise_filter_model,
        endpoint=ctx.noise_filter_endpoint,
    )
    if not verdict.permitted:
        logger.info("Skipping SOTA enrichment — %s", "; ".join(verdict.gaps))
    return verdict.permitted


def _stage_sota_pipeline(ctx: StageContext, state: StageState) -> StageState:
    """``sota_pipeline`` stage body — the composite front of the cloud arc.

    Each sub-phase (anonymize, entity_correction, sota_enrich,
    anon_plausibility) records its own block via phase_trace from inside
    ``_sota_pipeline``. ``_sota_pipeline`` reads ``chain_stopped()`` off the
    same contextvar a calibration caller's ``stop_at`` scope set, so it can
    short-circuit at any sub-phase boundary with no parameter threaded into
    it.

    Produces the seam the ``deanonymize`` and ``rebuild`` siblings consume:
    the surviving anonymized fact set, the cloud round-trip scope, the raw
    cloud response, the cloud's updated anonymized transcript, and the
    pre-pipeline relation count. Every early exit inside ``_sota_pipeline``
    (config bail, anonymize-failure divert, the empty-fact-set exit) hands
    back a state with an EMPTY ``facts``, which is exactly the spec's
    ``terminal_when`` — so the siblings do not run, matching the plain
    ``return graph`` those paths used to perform.

    Gated by ``ctx.validate`` plus :func:`_session_egress_permitted` (the
    shared cloud-admission verdict over ``sota_enabled`` / provider /
    model / key / endpoint — the flow's ``enabled_when``) and
    ``state.graph.relations`` being non-empty (the flow's
    ``applies_when``).
    """
    return _sota_pipeline(
        state.graph,
        ctx.transcript,
        ctx.model,
        ctx.tokenizer,
        provider=ctx.noise_filter,
        filter_model=ctx.noise_filter_model,
        endpoint=ctx.noise_filter_endpoint,
        plausibility_judge=ctx.plausibility_judge,
        plausibility_stage=ctx.plausibility_stage,
        plausibility_model=ctx.plausibility_model,
        plausibility_endpoint=ctx.plausibility_endpoint,
        speaker_name=ctx.speaker_name,
        speaker_id=ctx.speaker_id,
        scrub=ctx.scrub,
        correction_entity_types=ctx.correction_entity_types,
        prompts_dir=ctx.prompts_dir,
        model_alias=ctx.model_alias,
        max_tokens=ctx.max_tokens,
        plausibility_max_tokens=ctx.plausibility_max_tokens,
        seed=ctx.seed,
    )


def _stage_deanonymize(ctx: StageContext, state: StageState) -> StageState:
    """``deanonymize`` stage body — placeholders back to real names.

    Three things, in this order, because the order is load-bearing:

    1. The ``deanon`` phase: pure dict substitution restoring real names
       from placeholders (no LLM call, so ``raw_output`` stays ``None``).
       Facts still carrying an unresolved placeholder, and facts with a
       placeholder glued into the predicate, are dropped and recorded.
    2. The scalar partition. It is OWNED by ``rebuild``
       (``paramem.graph.relation_build``) but INVOKED here, before the
       judge below, and that placement is deliberate: scalars are URLs,
       emails, DOIs and version-tagged tool names, and the judge's drop
       rule R6 targets a "dot-separated or URI-shaped namespaced token"
       (``configs/prompts/sota_plausibility.txt``). Partitioning after
       the judge would expose exactly the values the partition protects.
       When the partition absorbs EVERY surviving fact (a scalar-only
       session) it records ``CAUSE_SCALAR_PARTITION`` — the ``routing``
       kind, which tells ``rebuild``'s recovery gate the resulting empty
       relation set is legitimate rather than a loss.
    3. The deanon-stage plausibility judge (local model, real names),
       which receives the ORIGINAL real-name transcript — it runs
       locally on de-anonymized facts, so there is no reason to hand it
       the anonymized text.

    The mid-stage ``chain_stopped()`` check after the ``deanon`` phase is
    the one the imperative version made: a calibration caller stopping at
    ``deanon`` must not get the partition or the judge.
    """
    graph = state.graph
    scope = state.scope
    facts = state.facts
    empty_cause = state.empty_cause

    # This is the SUBSTITUTION half of the two-call structure — the
    # ``sota_enrich`` phase's ``deanonymize_facts`` call was the GATE
    # (accept/reject decision, run before any plausibility filter could
    # shrink the fact set); this call substitutes whatever survived to
    # this point (post accept/reject AND post anon-stage plausibility, if
    # it ran).  Re-running the unconditional totality gate here is a
    # structurally guaranteed no-op on an already-accepted (and possibly
    # further-filtered, never further-expanded) subset — dropping facts
    # can only shrink the placeholder surface, never introduce a new
    # orphan — it is not a second privacy gap; it exists only because
    # ``deanonymize_facts`` is the ONE way to reach ``_apply_bindings``
    # (the structural guard).
    with phase_trace("deanon") as t:
        deanon_input_count = len(facts)
        deanon_result = deanonymize_facts(scope, facts)
        _record_binding_diagnostics(graph, deanon_result)
        deanon_facts = deanon_result.facts
        predicate_dropped = deanon_result.predicate_dropped
        residual_dropped = deanon_result.residual_dropped
        # predicate_dropped: facts SOTA returned with a placeholder glued
        # into the predicate field (_apply_bindings' step 1, pre-
        # substitution).  residual_dropped: facts still carrying an
        # unresolved placeholder after substitution (step 3).  The two
        # categories are returned already partitioned — no caller-side
        # recomputation.
        if predicate_dropped:
            graph.diagnostics["predicate_placeholder_dropped_facts"] = (
                graph.diagnostics.get("predicate_placeholder_dropped_facts", []) + predicate_dropped
            )
            graph.diagnostics["predicate_placeholder_dropped"] = graph.diagnostics.get(
                "predicate_placeholder_dropped", 0
            ) + len(predicate_dropped)
        if residual_dropped:
            graph.diagnostics["residual_dropped_facts"] = residual_dropped
        dropped_facts = predicate_dropped + residual_dropped
        if dropped_facts:
            logger.warning(
                "Dropped %d fact(s) post-substitution (%d predicate-invariant, "
                "%d residual placeholder sweep — composite string with an "
                "unresolved placeholder; a missing-binding orphan is rejected "
                "upstream by the whole-delta totality gate before reaching "
                "here).",
                len(dropped_facts),
                len(predicate_dropped),
                len(residual_dropped),
            )
        deanon_dropped = deanon_input_count - len(deanon_facts)
        if deanon_dropped:
            logger.info(
                "De-anon: %d → %d facts (%d dropped)",
                deanon_input_count,
                len(deanon_facts),
                deanon_dropped,
            )
        if deanon_input_count and not deanon_facts:
            empty_cause = CAUSE_DEANON_SUBSTITUTION
        t.set_parsed(
            {
                "input_count": deanon_input_count,
                "output_count": len(deanon_facts),
                "dropped_count": deanon_dropped,
                "dropped_facts": dropped_facts,
            }
        )
    if chain_stopped():
        # Calibration short-circuit: deanon recorded.  graph.relations
        # remains the local-extract output; deanonymized facts list is
        # in phases[deanon].parsed.
        return dataclasses.replace(state, graph=graph, facts=deanon_facts, empty_cause=empty_cause)

    # Route scalar-valued objects (URLs, emails, phone numbers, DOIs,
    # version-tagged tool names like "ROS2") off the relation surface and
    # onto Entity.attributes of the subject.  Scalars are verbatim
    # identifiers that flow through to plausibility and downstream filters
    # without modification.  Routing them to attributes mirrors the
    # email/phone/linkedin path the local extractor already populates and
    # which ``relation_prep._flatten_entity_attributes`` mints into keyed
    # pairs for indexed-key distillation.  The projection itself is applied
    # by the ``rebuild`` stage, after its entity rebuild, so the subject
    # entity survives pruning.
    scalar_facts, deanon_facts = partition_scalar_facts(deanon_facts, graph.entities)
    if scalar_facts:
        graph.diagnostics["scalar_facts_projected"] = len(scalar_facts)
        if not deanon_facts and empty_cause is None:
            # Scalar-only session: every surviving fact moved to the
            # attribute surface. ``empty_cause is None`` is what makes
            # this unambiguous — the substitution's own emptying is
            # recorded above and runs BEFORE the partition (so it would
            # leave nothing to partition), and the deanon judge below is
            # skipped on an empty fact set. A cause already recorded
            # means the emptying was mixed, and the recovery net keeps
            # its normal behaviour.
            empty_cause = CAUSE_SCALAR_PARTITION

    if state.sota_raw:
        graph.diagnostics["sota_raw_response"] = state.sota_raw
    if state.updated_anon_transcript:
        graph.diagnostics["sota_updated_transcript"] = state.updated_anon_transcript

    # Plausibility on de-anonymized data (local judge, stage="deanon").
    # Runs when plausibility_judge != "off" AND plausibility_stage == "deanon"
    # AND model/tokenizer are available (guard against tests that pass None).
    # "auto" resolves to the local model.
    if (
        ctx.plausibility_stage == "deanon"
        and ctx.plausibility_judge != "off"
        and deanon_facts
        and ctx.model is not None
        and ctx.tokenizer is not None
    ):
        with phase_trace("deanon_plausibility") as t:
            _vram_snapshot(f"before_plausibility_deanon session={graph.session_id}")
            filtered_deanon, plaus_raw = local_plausibility_filter(
                deanon_facts,
                ctx.transcript,  # original real-name transcript — intentional, see docstring
                ctx.model,
                ctx.tokenizer,
                max_tokens=ctx.plausibility_max_tokens,
                temperature=_DEFAULT_FILTER_TEMPERATURE,
                seed=ctx.seed,
                prompts_dir=ctx.prompts_dir,
            )
            t.set_raw(plaus_raw)
            if filtered_deanon is not None:
                pre_deanon = len(deanon_facts)
                deanon_facts = filtered_deanon
                dropped_deanon = pre_deanon - len(deanon_facts)
                graph.diagnostics["plausibility"] = "deanon"
                # Own key: the three plausibility writers (anon judge,
                # this one, and the raw-fallback judge) used to share
                # ``plausibility_dropped`` with three different
                # semantics, making its final value order-dependent.
                graph.diagnostics["plausibility_dropped_deanon"] = dropped_deanon
                graph.diagnostics["plausibility_judge_actual"] = (
                    ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                )
                if pre_deanon and not deanon_facts:
                    empty_cause = CAUSE_DEANON_JUDGE
                t.set_parsed(
                    {
                        "judge": (
                            ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                        ),
                        "input_count": pre_deanon,
                        "kept_count": len(deanon_facts),
                        "dropped_count": dropped_deanon,
                    }
                )
                logger.info(
                    "Deanon-stage plausibility (local): %d → %d facts (%d dropped)",
                    pre_deanon,
                    len(deanon_facts),
                    dropped_deanon,
                )
            else:
                t.set_outcome("failed", reason="plausibility parse returned None")
                t.set_parsed(
                    {
                        "judge": (
                            ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                        ),
                        "input_count": len(deanon_facts),
                        "kept_count": len(deanon_facts),
                        "dropped_count": 0,
                    }
                )
                logger.warning("Deanon-stage plausibility call failed — keeping deanon facts")

    return dataclasses.replace(
        state,
        graph=graph,
        facts=deanon_facts,
        scalar_facts=scalar_facts,
        empty_cause=empty_cause,
    )


def _stage_rebuild(ctx: StageContext, state: StageState) -> StageState:
    """``rebuild`` stage body — facts back to a ``SessionGraph``.

    Schema-validates the surviving fact dicts into ``Relation`` objects
    (recording every drop), consults the all-dropped recovery gate, and —
    when the gate does not fire — installs the relations together with
    their entity surface and the scalar-attribute projection. The pure
    half lives in :mod:`paramem.graph.relation_build`; what stays here is
    the recovery ACTION, which needs the model and tokenizer that module
    deliberately never sees.

    A scalar-only session reaches here with no relations and a
    ``routing`` cause: the gate declines, so ``apply_rebuild`` runs and
    the scalar projection lands on the entity surface. Suppressing the
    gate is what makes that possible — the recovery path returns before
    ``apply_rebuild``, and ``apply_rebuild`` is the only caller of the
    projection.
    """
    graph = state.graph
    kept_relations = build_relations(graph, state.facts, speaker_id=ctx.speaker_id)
    empty_cause = state.empty_cause
    if state.facts and not kept_relations:
        empty_cause = CAUSE_SCHEMA_VALIDATION

    # All-dropped safety net — if every relation was dropped and the
    # original extraction had facts, fall back to raw plausibility so the
    # session does not yield zero facts due to anonymizer inconsistency.
    if recovery_gate(graph, kept_relations, state.original_relation_count, empty_cause):
        return dataclasses.replace(
            state,
            graph=_fallback_plausibility_on_raw(
                graph,
                ctx.transcript,
                ctx.model,
                ctx.tokenizer,
                "all_dropped",
                speaker_name=ctx.speaker_name,
                speaker_id=ctx.speaker_id,
                max_tokens=ctx.max_tokens,
                plausibility_max_tokens=ctx.plausibility_max_tokens,
                seed=ctx.seed,
            ),
            empty_cause=empty_cause,
        )

    apply_rebuild(graph, kept_relations, state.scalar_facts, state.scope.resolution)

    added = len(kept_relations) - state.original_relation_count
    logger.info(
        "SOTA enrichment: %d → %d relations (%+d)",
        state.original_relation_count,
        len(kept_relations),
        added,
    )
    return dataclasses.replace(state, graph=graph, empty_cause=empty_cause)


SESSION_EXTRACT: list[StageSpec] = [
    StageSpec(
        stage="local_extract",
        trace_names=("local_extract",),
        run=_stage_local_extract,
        requires=frozenset(),
        produces=frozenset({"graph"}),
        terminal_when=lambda s: not s.graph.relations,
    ),
    StageSpec(
        stage="second_order_extract",
        trace_names=("second_order_extract",),
        run=_stage_second_order_extract,
        requires=frozenset({"graph"}),
        produces=frozenset({"graph"}),
        applies_when=lambda s: _has_named_non_speaker_person(s.graph),
    ),
    StageSpec(
        stage="sota_pipeline",
        # The composite opens four phases from inside ``_sota_pipeline``;
        # which of them fire depends on config (``anon_plausibility``
        # only when ``plausibility_stage == "anon"``).
        trace_names=("anonymize", "entity_correction", "sota_enrich", "anon_plausibility"),
        run=_stage_sota_pipeline,
        requires=frozenset({"graph"}),
        produces=frozenset(
            {
                "graph",
                "facts",
                "scope",
                "sota_raw",
                "updated_anon_transcript",
                "original_relation_count",
                "empty_cause",
            }
        ),
        enabled_when=lambda c: bool(c.validate) and _session_egress_permitted(c),
        applies_when=lambda s: bool(s.graph.relations),
        # Every exit that does NOT reach the hand-over point leaves
        # ``facts`` empty: the unsupported-provider and missing-config
        # bails, the anonymize-failure divert into the raw-plausibility
        # fallback, and the "nothing survived the anon-stage judge" exit
        # (which clears the graph itself before returning). All four used
        # to be a plain ``return graph`` from the middle of one long
        # function; here they stop the walk so the tail siblings never
        # run on a state that was never produced.
        terminal_when=lambda s: not s.facts,
    ),
    StageSpec(
        stage="deanonymize",
        # Two phases: the substitution itself, then the local judge at the
        # end of the same span (config-gated on ``plausibility_stage``).
        trace_names=("deanon", "deanon_plausibility"),
        run=_stage_deanonymize,
        requires=frozenset({"graph", "facts", "scope", "sota_raw", "updated_anon_transcript"}),
        produces=frozenset({"graph", "facts", "scalar_facts", "empty_cause"}),
        # A round-trip scope exists only once the composite reached its
        # hand-over. Without one there was no cloud egress at all — the
        # composite was disabled by config, or skipped because the graph
        # had no relations — and there is nothing to de-anonymize. The
        # composite's own ``terminal_when`` covers the other case (it RAN
        # but produced no surviving facts); ``applies_when`` cannot,
        # because a SKIPPED stage never stops the walk.
        applies_when=lambda s: s.scope is not None,
    ),
    StageSpec(
        stage="rebuild",
        # Pure post-processing plus the recovery action — no LLM phase of
        # its own. The fallback's plausibility call runs outside any
        # phase_trace scope, as it did before this runner existed.
        trace_names=(),
        run=_stage_rebuild,
        requires=frozenset(
            {"graph", "facts", "scalar_facts", "scope", "original_relation_count", "empty_cause"}
        ),
        produces=frozenset({"graph", "empty_cause"}),
        # Same gate as ``deanonymize``, and deliberately NOT ``bool(facts)``:
        # an empty fact set here is precisely the all-dropped case the
        # recovery gate inside this stage exists to catch.
        applies_when=lambda s: s.scope is not None,
    ),
]


def extract_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    speaker_id: str,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    prompts_dir: str | Path | None = None,
    validate: bool = True,
    noise_filter: str = "",
    noise_filter_model: str = "claude-sonnet-4-6",
    noise_filter_endpoint: str | None = None,
    sota_enabled: bool = False,
    speaker_name: str | None = None,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    plausibility_model: str = "claude-sonnet-4-6",
    plausibility_endpoint: str | None = None,
    *,
    scrub: set[str] | frozenset[str],
    correction_entity_types: set[str] | frozenset[str] | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
    model_alias: str | None = None,
    seed: int | None = None,
    timestamp: str | None = None,
    source_type: str = "transcript",
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Multi-pass pipeline:
    1. Extract candidate triples from transcript
    2. Second-order extraction: ``local_extract`` is speaker-centric, so a
       clause naming a non-speaker person by relationship ("my brother
       Nadeem lives in Porto") tends to keep only the speaker->person edge
       and drop that person's OWN fact (measured on Mistral 7B). This pass
       re-extracts facts ABOUT each named non-speaker person the first
       pass surfaced and unions them in; predicate drift or redundant
       re-emits are left to the existing merger dedup and normalization.
    3. SOTA pipeline (anonymize → enrich → anon-stage plausibility, configurable)
    4. De-anonymize (substitute real names back, scalar partition,
       deanon-stage plausibility)
    5. Rebuild (schema-validate relations, all-dropped recovery gate,
       entity surface + scalar-attribute projection)

    All filters fail gracefully — extraction result is preserved on any failure.

    A calibration caller may wrap this call in
    :func:`paramem.graph.phase_trace.stop_at` to return immediately after
    a named phase completes with a non-``"failed"`` outcome — saves
    compute when the operator only needs to inspect the early phases of
    the trace.  See :func:`~paramem.graph.phase_trace.stop_at`'s
    docstring for the mechanism; this function checks
    :func:`~paramem.graph.phase_trace.chain_stopped` after each phase
    block and returns early when it is set.  No scope open (the
    production default) means the full pipeline always runs.

    Args:
        temperature: Sampling temperature for extraction (default 0.0 for determinism).
        max_tokens: Max output tokens for extraction (default 2048).
        prompts_dir: Optional override for prompt config directory.
        validate: Run SOTA pipeline pass 3 (default True).
        noise_filter: SOTA provider for noise filtering ("" = disabled).
        plausibility_judge: Plausibility filter judge ("auto"=local, "off"=disabled,
            or a provider name from
            :data:`~paramem.utils.cloud_admission.PROVIDER_KEY_ENV` — e.g.
            "anthropic" — for cloud judging at anon stage).
        plausibility_stage: When to run plausibility ("deanon"=after de-anon,
            "anon"=on anonymized data with SOTA judge).
        plausibility_model: Model id the cloud judge runs. Ignored when
            ``plausibility_judge`` is "auto" or "off".
        plausibility_endpoint: Endpoint override for a self-hosted
            OpenAI-compatible judge. ``None`` accepts the provider's
            default; ignored for native-SDK providers.
        scrub: PII-vocabulary hints (``SanitizationConfig.scrub``) forwarded
            to the SOTA pipeline's local anonymizer call — the prompt is the
            sole scope authority (see :func:`anonymize_with_local_model`).
            Required — no implicit default; an empty ``set``/``frozenset``
            is the operator opt-out.
        correction_entity_types: Scope-and-enable knob for the local
            entity-surface correction stage (see
            :func:`paramem.graph.entity_correction.correct_entity_surfaces`).
            Entity-type members (``place``/``organization``/``concept``)
            gate which surfaces are corrected; ``"attributes"`` additionally
            enables correcting ``graph.entities[*].attributes`` values.
            ``None`` or empty disables the stage entirely — there is no
            implicit default scope; production always threads the
            configured value.
        speaker_id: Speaker store ID (e.g. ``"speaker0"``). Stamped onto every
            ``Relation`` produced by this extraction pass as provenance.
            Required — callers must always supply the session's speaker ID.
        system_prompt_filename: Filename of the system prompt within the prompts
            directory.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
            (``"extraction_system.txt"``).  Used for every source type;
            document chunks land in the ``{transcript}`` slot of the
            same prompt.
        user_prompt_filename: Filename of the user-turn prompt template.
            Defaults to :data:`DEFAULT_USER_PROMPT_FILENAME`
            (``"extraction.txt"``).  Same prompt for every source type.
        model_alias: Model alias (e.g. ``"qwen3-4b"``).  Enables per-file
            prompt resolution: ``prompts_dir/<model_alias>/<filename>``
            is checked first for local-model extraction prompts.  The
            ``sota_*`` prompts (cloud enricher) and ``anonymization.txt``
            are model-independent by design and are NOT affected by this
            parameter.
        seed: Optional RNG seed forwarded to every :func:`generate_answer`
            call within the pipeline (extraction, anonymization,
            plausibility).  At the default ``temperature=0.0`` (greedy
            decoding) this is a strict no-op.  Default ``None`` preserves
            production behaviour unchanged.
        timestamp: Session-start assertion time (ISO 8601), typically the
            session's ``started_at``.  Stamped onto the returned
            ``SessionGraph.timestamp`` so ``last_seen`` on newly-merged
            edges reflects when the facts were asserted, not when
            extraction ran.  ``None`` (default) falls back to ``now()``.
        source_type: ``"transcript"`` (default) or ``"document"``. Forwarded
            to :func:`_parse_extraction` / :func:`_stamp_speaker_entity` as
            the Guard B gate for the document-only exact-full-name rewrite
            of third-person speaker mentions onto ``speaker_id``.
    """
    # Open the extraction trace.  phase_trace() calls reachable from
    # any helper in this scope append to the same trace via contextvar;
    # the trace survives every `graph = ...` rebinding inside.  At the
    # end (any return path), trace.attach_to(graph) materialises the
    # records on the final graph's diagnostics.  A calibration caller may
    # additionally wrap this whole call in `with stop_at(phase):` (see
    # that function's docstring) to request an early return.
    #
    # The control flow (local_extract -> second_order_extract ->
    # sota_pipeline -> deanonymize -> rebuild) is expressed declaratively as
    # SESSION_EXTRACT and walked by paramem.graph.flow.run_flow — see that
    # module and the _stage_* functions above for the per-phase bodies.
    # This function's job is: build the run-constant StageContext once,
    # seed the initial StageState, walk the flow, and keep the
    # extraction_trace lifecycle (open/attach) exactly as before.
    with extraction_trace() as trace:
        state = StageState(
            graph=SessionGraph(
                session_id=session_id,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            )
        )
        try:
            ctx = StageContext(
                model=model,
                tokenizer=tokenizer,
                transcript=transcript,
                session_id=session_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                temperature=temperature,
                max_tokens=max_tokens,
                plausibility_max_tokens=plausibility_max_tokens,
                prompts_dir=prompts_dir,
                system_prompt_filename=system_prompt_filename,
                user_prompt_filename=user_prompt_filename,
                model_alias=model_alias,
                seed=seed,
                timestamp=timestamp,
                source_type=source_type,
                validate=validate,
                sota_enabled=sota_enabled,
                noise_filter=noise_filter,
                noise_filter_model=noise_filter_model,
                noise_filter_endpoint=noise_filter_endpoint,
                plausibility_judge=plausibility_judge,
                plausibility_stage=plausibility_stage,
                plausibility_model=plausibility_model,
                plausibility_endpoint=plausibility_endpoint,
                scrub=scrub,
                correction_entity_types=correction_entity_types,
            )
            state = run_flow(SESSION_EXTRACT, ctx, state)
            return state.graph
        finally:
            # Materialise the trace on whatever graph we're about to
            # return — covers every return path including early returns
            # on parse failure and empty-relations short-circuit.
            trace.attach_to(state.graph)


def extract_and_anonymize_for_cloud(
    transcript: str,
    model,
    tokenizer,
    *,
    speaker_id: str | None = None,
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    scrub: set[str] | frozenset[str],
) -> AnonymizedPayload:
    """Local extract + local anonymize for cloud egress.

    Composition over existing primitives — same anonymization chain
    ``_sota_pipeline`` runs every consolidation cycle, minus the SOTA
    enrichment call:

    0. **Turn-marking (model-facing only).** ``transcript`` — a bare,
       unmarked chat sentence at this call site — is rendered through
       :meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`
       (single source of truth for the ``[user] <text>`` /
       ``[assistant] <text>`` marker surface every extraction/
       anonymization few-shot is calibrated on) into
       ``model_facing_transcript``.  Only the two LLM calls below see the
       turn-marked copy — the marker exists solely to keep the model
       in-distribution while it authors the anonymized transcript. A bare
       sentence was observed to glue a possessive into a single
       anonymization token (``"Pat's dog"`` minted as one placeholder
       instead of splitting ``"Pat"`` + ``"dog"``) because it is
       off-distribution from every few-shot example.
    1. ``extract_graph(validate=False)`` — local extraction only,
       produces a SessionGraph the anonymizer can anchor on.
    2. :func:`~paramem.graph.cloud_egress.anonymize_for_cloud` — THE one
       anonymize chain, shared with every other cloud-egress path
       (identical to what :func:`_sota_pipeline` and
       :func:`~paramem.training.graph_enrich.run_graph_enrichment` call).
       ``identity_domain`` is not passed —
       this path has no closed node-key domain to reconcile against
       (free-text chat, not a fold graph).

    This helper anonymizes a TRANSCRIPT for cloud egress — it never
    builds or returns facts (the returned ``AnonymizedPayload.anon_facts``
    is always ``[]`` here, since ``graph.relations`` is discarded after
    anchoring the anonymizer — see step 1).

    ``scrub`` is the operator's PII-vocabulary hint list
    (``SanitizationConfig.scrub``), rendered verbatim into the
    anonymizer prompt's ``{scrub_categories}`` slot — the prompt is the
    SOLE scope authority; there is no code-side entity-type gate.
    Required — an omitted ``scrub`` would silently anonymize against a
    hidden default on a security-critical egress path.  An EMPTY
    ``scrub`` is the meaningful operator opt-out: it short-circuits
    before any LLM call (before even the local extraction pass — no
    compute is wasted anchoring an anonymizer that will never run), and
    the helper returns ``status="opted_out"`` with ``anon_transcript``
    sourced from the passed-in ``transcript`` verbatim, never a model
    artifact.

    ``speaker_id`` is the resolved speaker store ID, threaded to
    :func:`extract_graph` (which requires it) and stamped on the
    ephemeral graph's relations as provenance.  That graph exists only
    to anchor anonymization and is discarded immediately, so the value
    is never persisted.  Text-only ``/chat`` requests with no enrolled
    speaker pass ``None``; the helper falls back to the
    ``"cloud_egress"`` sentinel rather than failing extraction.

    ``AnonymizedPayload.status``:

    * ``"ok"`` — anonymization ran; ``anon_transcript`` is the MODEL's
      own rewrite (the single synthetic turn marker from step 0 stripped
      back off, restoring the bare-text contract this helper's callers
      expect).  ``forward``/``reverse`` may still be empty — a
      legitimate "ran, found nothing in scope" verdict, not a failure;
      egress PROCEEDS.
    * ``"opted_out"`` — operator opted out (``scrub`` empty).
    * ``"failed"`` — block.  Every other early exit lands here:
      empty/whitespace-only input, local extraction raising, zero
      relations extracted (``not graph.relations``), the anonymizer
      raising, an anonymizer parse failure, or the model's rewritten
      transcript coming back empty after the marker strip.  Callers must
      NEVER fall back to the original real-name transcript on this
      status.

    The companion :func:`~paramem.graph.cloud_egress.deanonymize_response_text`
    is the caller's exit gate for the cloud's response text.
    """
    # ``failure`` left unset (``None``) on this sentinel: it belongs to
    # ``anonymize_for_cloud``'s own "parse"/"guard" vocabulary (see
    # :class:`~paramem.graph.cloud_egress.AnonymizedPayload`), and none of
    # this helper's own early exits below (extraction exception, zero
    # relations, anonymizer exception) are that call's parse/guard
    # distinction — they're this helper's own failure modes, and no
    # caller here branches on cause.
    _failed = AnonymizedPayload(
        status="failed",
        forward={},
        reverse={},
        anon_transcript="",
        anon_facts=[],
        declared=frozenset(),
        norm_stats={"inverted": 0, "dropped": 0},
        rekey_dropped=0,
        raw="",
        failure=None,
    )
    if not transcript or not transcript.strip():
        return _failed

    # Empty scrub = operator opt-out.  Skip the entire LLM-driven
    # anonymization path (and the local extraction pass that would only
    # exist to anchor it) — the caller forwards the transcript verbatim.
    # Distinguished from the "failed" block status by non-empty text.
    if not scrub:
        return AnonymizedPayload(
            status="opted_out",
            forward={},
            reverse={},
            anon_transcript=transcript,
            anon_facts=[],
            declared=frozenset(),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
            failure=None,
        )

    # The local_extract and anonymization few-shots (configs/prompts/
    # extraction.txt, configs/prompts/anonymization.txt) are calibrated
    # exclusively on the ``[user] <text>`` / ``[assistant] <text>``
    # turn-marked surface produced by ``SessionBuffer._format_turns`` (the
    # single source of truth for that rendering — see its docstring,
    # which already names this call site as the intended second caller).
    # A bare, unmarked sentence puts the model off-distribution from
    # every example it was tuned on and has been observed to glue a
    # possessive into a single anonymization token (e.g. "Pat's dog" ->
    # one placeholder instead of "Pat" + "dog" separately).
    turn_marked_lines, _ = SessionBuffer._format_turns([{"role": "user", "text": transcript}])
    model_facing_transcript = "\n".join(turn_marked_lines)
    # The single synthetic turn always renders as ``"<marker> " + transcript``
    # (see ``_format_turns``) — derive the marker prefix mechanically rather
    # than hardcoding the literal, so a future marker-format change (single
    # source of truth: ``_format_turns``) cannot silently desync this strip.
    _turn_marker_prefix = model_facing_transcript[: len(model_facing_transcript) - len(transcript)]

    # Both LLM calls below (extraction + anonymization) are structured
    # extraction and must run on the base weights, never the training-active
    # adapter.  One shared scope disables the adapter and keeps the KV cache
    # live for both generates, restoring the model's entry state on exit.
    with base_model_inference(model):
        try:
            graph = extract_graph(
                model,
                tokenizer,
                model_facing_transcript,
                session_id="cloud_egress",
                # Ephemeral graph: extracted only to anchor anonymization, then
                # discarded — the stamped provenance is never persisted.  Use the
                # resolved speaker_id when the caller has one (text-only /chat
                # requests may not), falling back to the "cloud_egress" sentinel
                # that already names the session.
                speaker_id=speaker_id or "cloud_egress",
                speaker_name=speaker_name,
                # No model_alias: cloud-egress extraction is deliberately
                # model-independent.  This graph exists only to anchor PII
                # anonymization (entity spans), not to build the knowledge graph,
                # and entity/attribute extraction is reliable across models — the
                # per-model prompt overrides target relation decomposition, which
                # is not load-bearing here.  So the shared base prompt is used
                # regardless of the configured model.
                prompts_dir=prompts_dir,
                validate=False,
                noise_filter="",
                scrub=scrub,
            )
        except Exception:
            logger.exception("Cloud egress: local extraction failed; treating as block")
            return _failed

        if not graph.relations:
            return _failed

        try:
            payload = anonymize_for_cloud(
                graph,
                model,
                tokenizer,
                transcript=model_facing_transcript,
                scrub=scrub,
                speaker_name=speaker_name,
                prompts_dir=prompts_dir,
                max_tokens=_DEFAULT_ANONYMIZER_MAX_TOKENS,
            )
        except Exception:
            logger.exception("Cloud egress: anonymization raised; treating as block")
            return _failed

    if payload.status == "failed":
        return payload

    # The returned transcript is the MODEL's own rewrite — never
    # mechanically rebuilt from the mapping.  Strip the single synthetic
    # turn marker (step 0) back off so the bare-text contract this
    # helper's callers rely on is preserved; best-effort (the model is
    # instructed to preserve everything outside the substitutions
    # verbatim, including the marker, but a strip that finds nothing to
    # strip is a no-op).
    anon_transcript = payload.anon_transcript
    if _turn_marker_prefix and anon_transcript.startswith(_turn_marker_prefix):
        anon_transcript = anon_transcript[len(_turn_marker_prefix) :]

    if not anon_transcript:
        return _failed

    return dataclasses.replace(payload, anon_transcript=anon_transcript)


def _generate_extraction(
    model,
    tokenizer,
    transcript: str,
    temperature: float,
    max_tokens: int,
    prompts_dir: str | Path | None = None,
    speaker_name: str | None = None,
    *,
    speaker_id: str | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
    model_alias: str | None = None,
    seed: int | None = None,
    vram_label: str = "extract_main",
) -> str:
    """Generate graph extraction output from the model. Called once.

    Narrator binding is achieved via the ``{speaker_context}`` placeholder
    in the **user** template (``extraction.txt``), populated by
    :func:`build_speaker_context`.  The directive pins the stable
    ``speaker_id`` (e.g. ``"speaker0"``) as the subject of every extracted
    speaker-fact, with the display ``speaker_name`` supplied as comprehension
    context so the model maps self-references onto the id.

    The system prompt is passed verbatim — no slot substitution is performed
    on it.  One prompt-pair serves every source type — document chunks land
    in the same ``{transcript}`` slot.

    ``model_alias`` enables per-file prompt resolution — see
    :func:`load_extraction_prompts`.  The ``sota_*`` prompts are
    model-independent by design and are not affected.

    ``seed`` is forwarded verbatim to :func:`generate_answer`.  At the
    default ``temperature=0.0`` (greedy decoding) it is a strict no-op.

    ``vram_label`` names the :func:`~paramem.server.vram_guard.vram_scope`
    wrap around the generate call. Defaults to ``"extract_main"``
    (``local_extract``/``second_order_extract``); callers with a distinct
    VRAM-telemetry identity (e.g. ``extract_procedural_graph`` using
    ``"procedural"``) override it.
    """
    system, prompt = load_extraction_prompts(
        prompts_dir,
        system_filename=system_prompt_filename,
        user_filename=user_prompt_filename,
        model=model_alias,
    )
    speaker_context = build_speaker_context(speaker_id, speaker_name)
    format_kwargs = dict(
        transcript=transcript,
        speaker_context=speaker_context,
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": prompt.format(**format_kwargs),
        },
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer), tokenize=False, add_generation_prompt=True
    )

    # vram_scope: main extraction generate is the longest prefill of the
    # extraction chain. Without an empty_cache between this phase and the
    # downstream anonymization / entity_correction / plausibility prefills,
    # the ``past_key_values`` from this generate stay pinned and compound
    # into the next phase's allocation. Symmetric with the
    # ``vram_scope("entity_correction")`` and ``vram_scope("plaus_filter")``
    # wraps elsewhere in the chain.
    with vram_scope(vram_label):
        return generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )


def _parse_extraction(
    raw_output: str,
    session_id: str,
    speaker_id: str,
    speaker_name: str | None = None,
    timestamp: str | None = None,
    source_type: str = "transcript",
) -> SessionGraph:
    """Parse raw model output into a SessionGraph.

    Handles non-standard field names, array-valued fields, and other
    model output quirks via _normalize_extraction. Local models occasionally
    emit a bare JSON array of fact dicts instead of the expected
    ``{"entities": [...], "relations": [...]}`` envelope; that case is
    rewrapped here so downstream normalization can proceed.

    After schema validation, :func:`_stamp_speaker_entity` is called to stamp
    ``speaker_id`` on every entity whose name is a speaker id (``speaker{N}``
    format).  The session speaker receives the authoritative lowercase
    ``speaker_id`` value; other speaker-id entities receive their own name as
    ``speaker_id``.

    Args:
        raw_output: Raw model output string.
        session_id: Session identifier for the graph.
        speaker_id: Speaker store ID (e.g. ``"speaker0"``).  Stamped onto
            every relation as provenance and used to identify the session
            speaker entity.  Required — callers must always supply a real id.
        speaker_name: Display name of the speaker (e.g. ``"Alex"``).  Used
            for document-source exact-full-name binding in
            :func:`_stamp_speaker_entity` (fires only when ``source_type ==
            "document"`` and the name is multi-token — see that function's
            guards); has no effect on transcript-source extraction.
        timestamp: Session-start assertion time (ISO 8601), forwarded from
            the caller's ``extract_graph(timestamp=...)``.  ``None`` (default)
            falls back to ``now()``.
        source_type: ``"transcript"`` (default) or ``"document"``. Forwarded
            to :func:`_stamp_speaker_entity` as the Guard B gate for the
            exact-full-name rewrite.
    """
    json_str = _extract_json_block(raw_output)
    data = json.loads(json_str)

    if isinstance(data, list):
        # Bare list of facts — wrap as a relations payload. _normalize_extraction
        # walks ``relations`` and infers the entity set from subject/object.
        data = {"relations": data, "entities": []}
    elif not isinstance(data, dict):
        raise ValueError(f"Unexpected extraction payload type: {type(data).__name__}")

    data["session_id"] = session_id
    data["timestamp"] = timestamp or datetime.now(timezone.utc).isoformat()

    data = _normalize_extraction(data)

    # Stamp speaker_id onto every relation dict before schema validation.
    # Relation.speaker_id is mandatory; the LLM output never includes it.
    for rel_dict in data.get("relations", []):
        rel_dict.setdefault("speaker_id", speaker_id)

    graph = SessionGraph.model_validate(data)

    # Post-process: stamp speaker_id on speaker-id entities.
    if speaker_id:
        graph = _stamp_speaker_entity(
            graph, speaker_id=speaker_id, speaker_name=speaker_name, source_type=source_type
        )

    logger.info(
        "Extracted graph: %d entities, %d relations (session=%s)",
        len(graph.entities),
        len(graph.relations),
        session_id,
    )
    return graph


def _stamp_speaker_entity(
    graph: SessionGraph,
    *,
    speaker_id: str,
    speaker_name: str | None = None,
    source_type: str = "transcript",
) -> SessionGraph:
    """Stamp ``speaker_id`` on speaker-id entities in the extracted graph.

    Under the id-as-subject convention the extraction prompt instructs the
    model to emit ``speaker{N}`` (lowercase) as the entity name and relation
    subject.  The ingest safety-net in :func:`_normalize_extraction` ensures
    the entity name is already lowercase before this function is called.

    Two responsibilities compose here, in order:

    1. **Exact-full-name rewrite (document sources only).** Third-person
       documents (e.g. a CV: "Alex Walker led the team") describe the
       speaker by literal name rather than self-reference, so the model has
       no ``speaker{N}`` token to emit for the subject.  When both guards
       below hold, every entity/relation field that matches the speaker's
       full name via :func:`~paramem.utils.identity.canonical` (exact
       match only — no substrings, no first-name subsets, no honorifics) is
       rewritten to ``canonical(speaker_id)`` *before* the stamping loop
       below runs, so the rewritten entities are also stamped.

       * **Guard A (full-name only)** — fires only when
         ``len(canonical(speaker_name).split("_")) >= 2``.  ``canonical``
         emits ``_`` as the blank, so the name parts are the ``_``-separated
         tokens of the canonical form; ``-`` is not a blank, so a hyphenated
         single given name ("Anna-Maria") counts as ONE part. A single-token
         display name (``resolve_speaker_name`` routinely returns a bare
         first name) fails closed: no rewrite. This is what prevents a
         first-person transcript mention like "My friend Alex came over"
         from ever being eligible (paired with Guard B below, which excludes
         transcripts outright).
       * **Guard B (document sources only)** — fires only when
         ``source_type == "document"``. Transcript/voice sessions keep
         today's first-person comprehension binding untouched.

       Rewriting a relation's ``object`` to the speaker id can synthesize a
       ``(speaker0, pred, speaker0)`` self-loop when the same full name was
       both subject and object (e.g. "X reported to X" narration collapsing
       onto one person). The self-loop filter in :func:`_normalize_extraction`
       already ran earlier in the pipeline and cannot see a self-loop this
       rewrite creates, and the graph merger has no ``subject == object``
       guard, so such relations are dropped here — the whole relation, not
       just the object field, since leaving the literal name on ``object``
       would both re-expose it to downstream anonymization and produce a
       bogus fact.

       The rewrite can also produce two entities named ``canonical(speaker_id)``
       in one graph (a pre-existing speaker entity plus one renamed by the
       rewrite); those are collapsed into one, unioning ``attributes``.

    2. **Existing ``is_speaker_id`` stamping loop** (unchanged in shape) sets
       ``entity.speaker_id`` on every entity whose (possibly just-rewritten)
       name passes :func:`~paramem.utils.identity.is_speaker_id`:

       * **Session speaker** — when ``ent.name == canonical(speaker_id)`` the
         entity IS the session speaker.  Its ``speaker_id`` field is set to
         ``canonical(speaker_id)`` (the authoritative lowercase id from the
         caller), preserving the authoritative-id pin.  This guards against
         the model emitting the wrong digit (e.g. ``speaker1`` in a
         ``speaker0`` session).

       * **Other speaker reference** — when a different ``speaker{N}`` id
         appears (e.g. a third-party speaker), ``entity.speaker_id`` is set
         to ``ent.name`` (already lowercase via the ingest safety-net).

    Non-speaker entities (display names like ``"Jordan Becker"``, places, orgs)
    are left untouched: ``is_speaker_id`` returns ``False`` for them, and the
    rewrite in (1) only fires under both guards.

    Args:
        graph: Parsed :class:`~paramem.graph.schema.SessionGraph` after
            schema validation.
        speaker_id: Authoritative speaker id (e.g. ``"speaker0"``).  Used as
            the authoritative-pin guard to detect wrong-digit model emissions,
            and as the rewrite target in (1).
        speaker_name: Display name of the speaker (e.g. ``"Alex Walker"``),
            used ONLY as the exact-full-name rewrite target in (1). ``None``
            (default; e.g. an anonymous speaker) skips the rewrite — there is
            nothing to bind a third-person mention to.
        source_type: ``"transcript"`` (default) or ``"document"``. The
            rewrite in (1) fires only for ``"document"`` (Guard B).

    Returns:
        Updated ``SessionGraph`` with ``speaker_id`` set on all speaker-id
        entities, and (for document sources with a full display name) the
        speaker's literal full name rewritten to ``canonical(speaker_id)``
        wherever it appears verbatim as an entity name or relation
        subject/object. Entities/relations that don't match are unmodified.
    """
    if not graph.entities:
        return graph

    # Multi-part-name gate: the rewrite only fires for a full name (given +
    # family), never a bare given name.  ``canonical`` emits ``_`` as the blank,
    # so the name parts are the ``_``-separated tokens of the canonical form.
    # ``-`` is not a blank, so a hyphenated single given name ("Anna-Maria")
    # counts as ONE part and does not open the rewrite.
    do_rewrite = (
        source_type == "document"
        and speaker_name
        and not is_speaker_id(speaker_name)
        and len(canonical(speaker_name).split("_")) >= 2
    )

    # The authoritative speaker id in its canonical form.  Computed once and
    # shared by the rewrite block and the stamping loop below — both need the
    # identical value, and identity strings route through ``canonical``.
    sid = canonical(speaker_id)

    if do_rewrite:
        target = canonical(speaker_name)

        for ent in graph.entities:
            if canonical(ent.name) == target:
                ent.name = sid
        for rel in graph.relations:
            if canonical(rel.subject) == target:
                rel.subject = sid
            if canonical(rel.object) == target:
                rel.object = sid

        # Self-loop drop: dropping the whole relation (not just skipping the
        # object rewrite) is the only choice that keeps the literal name out
        # of anonymization — see docstring point (1).
        kept_relations = []
        for rel in graph.relations:
            if rel.subject == sid and rel.object == sid:
                logger.debug("Filtered rewrite-synthesized self-loop: %s -> %s", sid, sid)
                continue
            kept_relations.append(rel)
        graph.relations = kept_relations

        # Duplicate-speaker collapse: keep the first speaker_id-named entity,
        # union attributes from any later duplicate the rewrite produced.
        kept_entities = []
        speaker_entity = None
        for ent in graph.entities:
            if ent.name == sid:
                if speaker_entity is None:
                    speaker_entity = ent
                    kept_entities.append(ent)
                else:
                    speaker_entity.attributes = {**ent.attributes, **speaker_entity.attributes}
                continue
            kept_entities.append(ent)
        graph.entities = kept_entities

    for ent in graph.entities:
        if not is_speaker_id(ent.name):
            continue
        if ent.name == sid:
            # This entity IS the session speaker — stamp the authoritative id.
            # Both values are canonical; the explicit pin preserves the
            # authoritative-id guard (model may emit wrong digit).
            ent.speaker_id = sid
        else:
            # A different registered speaker referenced in this session.
            # ent.name is already lowercase via the ingest safety-net.
            ent.speaker_id = ent.name

    return graph


# _JSON_ENVELOPE_KEYS / _extract_json_block now live in
# paramem.graph.cloud_egress (imported above) — every SOTA-response
# parser in this module still routes through the one shared parser.


# Fallbacks resolved per-call via schema_config.


def _normalize_extraction(data: dict) -> dict:
    """Normalize model output to match SessionGraph schema.

    Handles common field name variations from free-form generation.
    """
    # Normalize entities
    if "entities" in data:
        normalized_entities = []
        for ent in data["entities"]:
            if not isinstance(ent, dict):
                continue
            norm = {}
            raw_name = ent.get("name") or ent.get("entity", "unknown")
            if isinstance(raw_name, list):
                raw_name = raw_name[0] if raw_name else "unknown"
            norm["name"] = str(raw_name).strip()
            # Ingest safety-net: canonicalize any speaker-id token at the single
            # normalization boundary.  Extraction prompts instruct the model to
            # emit lowercase speaker{N} directly; this coerces any residual cased
            # form (e.g. "Speaker0") a model emits despite the instruction.
            # ONLY speaker-id tokens are coerced — display names are untouched.
            if is_speaker_id(norm["name"]):
                norm["name"] = canonical(norm["name"])
            raw_type = ent.get("entity_type") or ent.get("type", "concept")
            if isinstance(raw_type, list):
                raw_type = raw_type[0] if raw_type else "concept"
            fb_etype = fallback_entity_type()
            # entity_type is open (no Literal enforcement) — accept any
            # non-empty string so the model can emit rich types like
            # "product", "certification", "program", "paper", etc.
            # The schema YAML's entity_types list is a soft prior for
            # prompt examples; it does not gate the value here.
            type_str = canonical(str(raw_type)) if raw_type else ""
            norm["entity_type"] = type_str if type_str else fb_etype
            raw_attrs = ent.get("attributes", {})
            if not isinstance(raw_attrs, dict):
                raw_attrs = {}
            # Filter None values — model often outputs {"age": null}
            norm["attributes"] = {k: str(v) for k, v in raw_attrs.items() if v is not None}
            # If model put extra fields as top-level, capture them as strings
            skip_keys = {"name", "entity", "entity_type", "type", "attributes"}
            for k, v in ent.items():
                if k not in skip_keys and v is not None:
                    norm["attributes"][k] = str(v)
            normalized_entities.append(norm)
        data["entities"] = normalized_entities

    # Normalize relations
    if "relations" in data:
        # Expand multi-object relations: {"objects": ["A", "B"]} → two relations
        expanded = []
        for rel in data["relations"]:
            if not isinstance(rel, dict):
                continue
            objects = rel.get("objects")
            if isinstance(objects, list) and "object" not in rel:
                for obj_val in objects:
                    new_rel = {k: v for k, v in rel.items() if k != "objects"}
                    new_rel["object"] = obj_val
                    expanded.append(new_rel)
            else:
                expanded.append(rel)

        normalized_relations = []
        for rel in expanded:
            raw_subj = rel.get("subject") or "unknown"
            raw_obj = rel.get("object") or "unknown"
            if isinstance(raw_subj, list):
                raw_subj = raw_subj[0] if raw_subj else "unknown"
            if isinstance(raw_obj, list):
                raw_obj = raw_obj[0] if raw_obj else "unknown"
            subject = str(raw_subj).strip()
            obj = str(raw_obj).strip()
            # Ingest safety-net: canonicalize any speaker-id token at this boundary.
            if is_speaker_id(subject):
                subject = canonical(subject)
            if is_speaker_id(obj):
                obj = canonical(obj)

            # Filter self-loops (e.g. "KIT studied at KIT").  This is the sole
            # self-loop guard; the merger has none.  Comparing canonical forms
            # makes the guard agree exactly with the merger's node-key function,
            # so any pair that WOULD land on one node key is rejected here
            # instead of becoming a self-edge: it additionally catches
            # diacritic ("José"/"Jose"), full-case-fold ("Straße"/"Strasse") and
            # blank-run ("New York"/"New_York") variants that .lower() missed.
            # "-" is not a blank, so "Anna-Maria" and "Anna Maria" stay distinct
            # endpoints and are NOT filtered.
            if canonical(subject) == canonical(obj):
                logger.debug("Filtered self-loop: %s -> %s", subject, obj)
                continue

            raw_confidence = rel.get("confidence", 1.0)
            try:
                raw_confidence = float(raw_confidence)
            except (TypeError, ValueError):
                raw_confidence = 1.0
            # Model may use 0-100 scale instead of 0-1
            if raw_confidence > 1.0:
                raw_confidence = raw_confidence / 100.0
            norm = {
                "subject": subject,
                "predicate": (rel.get("predicate") or "related_to").strip(),
                "object": obj,
                "confidence": max(0.0, min(1.0, raw_confidence)),
            }
            raw_type = rel.get("relation_type") or rel.get("type", "factual")
            fb_rtype = fallback_relation_type()
            norm["relation_type"] = raw_type if raw_type in set(relation_types()) else fb_rtype
            # Preserve speaker_id if already present (stamped upstream).
            # Production code stamps it after _normalize_extraction; round-trip
            # paths (tests, restore flows) may supply it in the raw dict.
            if "speaker_id" in rel:
                norm["speaker_id"] = rel["speaker_id"]
            normalized_relations.append(norm)
        data["relations"] = normalized_relations

    # Ensure required top-level fields, coerce None to defaults
    if data.get("summary") is None:
        data["summary"] = ""
    data.setdefault("summary", "")
    data.setdefault("entities", [])
    data.setdefault("relations", [])

    return data


# Two-stage SOTA pipeline: enrichment first, then plausibility filtering.
# Each stage has a single responsibility and a separate prompt — combining
# them in one call (the previous "noise_filter" prompt) led to the LLM
# expanding scope at the same time as filtering, producing inflated counts
# and self-referential schema artifacts.

# The cloud plausibility judge — a judge that only ever sees anonymized
# data — dispatches on the SAME provider registry as enrichment
# (PROVIDER_KEY_ENV in paramem.utils.cloud_admission).  ``plausibility_judge``
# names a provider from that registry; ``plausibility_model`` and
# ``plausibility_endpoint`` name the model and (for a self-hosted
# OpenAI-compatible host) the URL.  "auto" and "off" are deliberately NOT
# providers — they select the local deanon-stage judge and no judge
# respectively, and membership in PROVIDER_KEY_ENV is what keeps them from
# ever reaching a key lookup.


def _fallback_plausibility_on_raw(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    reason: str,
    *,
    speaker_name: str | None = None,
    speaker_id: str,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    seed: int | None = None,
) -> SessionGraph:
    """Fallback pipeline path: run local plausibility on raw (unanonymized) facts.

    Used when anonymization fails entirely (mapping parse failure — no safe
    SOTA path), or when the full pipeline drops all relations.

    Steps (originally ported from a retired standalone comparison script):
    1. Serialize graph.relations to fact dicts. These are already
       real-name, un-anonymized ``graph.relations`` — no placeholder
       vocabulary exists at this point (nothing was ever anonymized on
       this path), so there is nothing to sweep here.
    2. If non-empty, run local plausibility filter; keep raw on None return.
    3. Rebuild Relations via :func:`~paramem.graph.relation_build.build_relations`
       — the ONE ``Relation``-construction site — and filter entities down
       to the surviving endpoints.
    4. Record fallback_path in diagnostics.

    Step 3 shares ``build_relations`` with the ``rebuild`` stage, so a
    schema-validation failure on this path lands in
    ``graph.diagnostics["pydantic_validation_dropped"]`` like anywhere
    else. This path used to swallow those failures with a bare
    ``except: continue``, which made a recovery-path validation failure
    invisible.

    Args:
        speaker_id: Speaker store ID stamped onto every reconstructed
            ``Relation`` as provenance. Required — callers must always supply
            the session's speaker ID.

    Returns the modified graph in-place (graph.relations / graph.entities replaced).
    """
    raw_facts = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
            "symmetric": getattr(r, "symmetric", False),
        }
        for r in graph.relations
    ]

    # Local plausibility filter (uses real names).
    if raw_facts and model is not None and tokenizer is not None:
        filtered, _raw = local_plausibility_filter(
            raw_facts,
            transcript,
            model,
            tokenizer,
            max_tokens=plausibility_max_tokens,
            temperature=_DEFAULT_FILTER_TEMPERATURE,
            seed=seed,
        )
        if filtered is not None:
            pre = len(raw_facts)
            raw_facts = filtered
            dropped_count = pre - len(raw_facts)
            if dropped_count:
                # Own key: this judge, the anon-stage judge and the
                # deanon-stage judge used to share
                # ``plausibility_dropped``, so whichever ran last decided
                # what the number meant.
                graph.diagnostics["plausibility_dropped_fallback"] = dropped_count
                graph.diagnostics["plausibility_judge_actual"] = "local_fallback"

    # Rebuild Relations from surviving raw facts.
    kept_relations = build_relations(graph, raw_facts, speaker_id=speaker_id)
    kept_names = {r.subject for r in kept_relations} | {r.object for r in kept_relations}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    graph.relations = kept_relations

    # Record fallback path in diagnostics.
    graph.diagnostics["fallback_path"] = reason
    logger.info(
        "_fallback_plausibility_on_raw: reason=%r, %d relation(s) surviving",
        reason,
        len(kept_relations),
    )
    return graph


def _record_binding_diagnostics(graph: SessionGraph, result: DeanonResult) -> None:
    """Persist a :func:`~paramem.graph.cloud_egress.deanonymize_facts`
    verdict onto ``graph.diagnostics`` — the CALLER side of the totality
    gate.

    The gate primitive
    (:func:`~paramem.graph.placeholders._check_mapping_totality`) used to
    write these two keys itself, from inside ``deanonymize_facts``, onto a
    ``SessionGraph`` it took purely as a diagnostics sink; a caller two
    levels up then read the mutation back off the graph. Both findings are
    return values now, and this is the ONE place in the extractor that
    turns them into diagnostics, shared by all three
    ``deanonymize_facts`` call sites (the ``sota_enrich`` gate, the
    ``deanon`` substitution, and :func:`_graph_enrich_with_sota`).

    Writes are guarded exactly as the primitive's were: an EMPTY list
    writes no key at all, so ``"key" not in graph.diagnostics`` keeps its
    established meaning ("the scan found nothing"), distinct from a
    present-but-empty value.

    Args:
        graph: The graph the delta is being applied to — the session graph
            for session-tier extraction, the caller's throwaway per-chunk
            graph for graph-tier enrichment.
        result: The ``DeanonResult`` just returned for that delta.
    """
    if result.collisions:
        graph.diagnostics["sota_binding_collisions"] = result.collisions
    if result.verdict:
        graph.diagnostics["sota_pending_orphans"] = result.verdict


def _sota_pipeline(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    speaker_id: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    endpoint: str | None = None,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    plausibility_model: str = "claude-sonnet-4-6",
    plausibility_endpoint: str | None = None,
    speaker_name: str | None = None,
    *,
    scrub: set[str] | frozenset[str],
    correction_entity_types: set[str] | frozenset[str] | None = None,
    prompts_dir: str | Path | None = None,
    model_alias: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    seed: int | None = None,
) -> StageState:
    """Enrich extraction via local anonymization → SOTA enrichment → anon-stage plausibility.

    ``scrub`` (required — no implicit default) is the operator's
    PII-vocabulary hint list, rendered into the anonymizer prompt's
    ``{scrub_categories}`` slot — the model is the sole scope authority.
    An EMPTY ``scrub`` is the operator opt-out: no anonymizer
    call is made; the transcript egresses verbatim (sourced from the
    passed-in ``transcript``, never a model artifact) and facts pass
    through ``_build_anon_facts`` with an empty (identity) mapping.

    Stages (non-empty ``scrub``):
    1. Local anonymize    → the model is the SOLE scope authority: it
       classifies real values against ``scrub`` and returns BOTH the
       real→placeholder mapping AND its own rewrite of the transcript with
       those values placeholdered (``anonymized_transcript``).  The
       script still builds the anonymized FACT array deterministically from
       ``graph.relations`` and the mapping (subject/object substituted,
       predicate/relation_type/confidence copied verbatim) — facts are
       never taken from the model's response.  Fail-closed: a parse
       failure or missing/empty ``anonymized_transcript`` falls back to
       local plausibility on the LOCAL-EXTRACT facts, never to the
       original real-name transcript.
    1c. Local entity-surface correction (:func:`paramem.graph.entity_correction.
        correct_entity_surfaces`) — corrects misspelled real place/org/concept
        surfaces in the reverse map AND, when configured, ``graph.entities[*].
        attributes`` values; forward map + transcript are untouched (see
        ``entity_correction.py`` for the full two-locus contract).
    2. SOTA enrichment    → coreference resolution + compound splitting + symmetric dedup
    3. Plausibility on anonymized data (plausibility_stage="anon", SOTA judge)

    De-anonymization, the deanon-stage judge and the relation/entity
    rebuild are NOT here: they are the ``deanonymize`` and ``rebuild``
    stages of ``SESSION_EXTRACT``, siblings of this composite. This
    function's job ends at the hand-over.

    Returns a :class:`~paramem.graph.flow.StageState` carrying what those
    siblings consume: the surviving anonymized ``facts``, the cloud
    round-trip ``scope``, ``sota_raw``, ``updated_anon_transcript`` and
    ``original_relation_count``. Every early exit returns a state with an
    EMPTY ``facts`` — the composite spec's ``terminal_when`` — so the
    siblings do not run, which is exactly what the plain ``return graph``
    on those paths used to achieve.

    Falls back gracefully at every stage. Endpoint is forwarded for self-hosted
    OpenAI-compatible providers.

    Plausibility judges:
    - "auto"  → local model at deanon stage (zero cloud cost, privacy-safe)
    - "off"   → disable plausibility entirely
    - any provider name in
      :data:`~paramem.utils.cloud_admission.PROVIDER_KEY_ENV`
      (``"anthropic"``, ``"openai"``, ``"google"``, …) → cloud judge at
      anon stage, running ``plausibility_model`` at
      ``plausibility_endpoint`` (must be combined with
      ``plausibility_stage="anon"`` to avoid PII exfiltration — the server
      config rejects the ``"deanon"`` pairing at load time)

    A calibration :func:`~paramem.graph.phase_trace.stop_at` request is
    read off :func:`~paramem.graph.phase_trace.chain_stopped` — a
    contextvar set by whichever caller (if any) opened a ``stop_at``
    scope around this call, rather than a parameter threaded into this
    function.
    """
    # Admission gate — shared with graph-tier enrichment, predicate
    # normalization and /calibrate/enrich.  ``sota_enabled=True`` is the
    # literal truth at this point: the master switch is the flow's
    # ``enabled_when`` term (:func:`_session_egress_permitted`), which is
    # the only way this function is reached in production.  Re-checking
    # the remaining terms here keeps the composite safe for a direct
    # caller that has no StageContext.
    verdict = evaluate_cloud_egress(
        sota_enabled=True, provider=provider, model=filter_model, endpoint=endpoint
    )
    if not verdict.permitted:
        logger.info("Skipping SOTA enrichment — %s", "; ".join(verdict.gaps))
        return StageState(graph=graph)
    api_key = verdict.api_key
    endpoint = verdict.endpoint

    original_count = len(graph.relations)
    # Which site emptied the working fact set, if any. Recorded onto the
    # returned state for the recovery gate's diagnostics; nothing branches
    # on it. See paramem.graph.relation_build for the vocabulary.
    empty_cause: str | None = None

    _vram_snapshot(f"sota_pipeline_entry session={graph.session_id}")
    if not scrub:
        # Operator opt-out: no anonymizer call, no phase trace (mirrors
        # the pre-unification structure exactly — including that a
        # ``stop_at("anonymize")`` request does NOT short-circuit here,
        # since there is nothing to stop after: the "anonymize" phase
        # never fires, so ``chain_stopped()`` can never become true from
        # it on this branch).  The transcript egresses verbatim, sourced
        # from the passed-in transcript — never a model artifact.  Facts
        # follow via the (empty-mapping, identity) ``_build_anon_facts``
        # call below.
        payload = anonymize_for_cloud(
            graph, model, tokenizer, transcript=transcript, scrub=scrub, speaker_name=speaker_name
        )
        graph.diagnostics["anonymize"] = "opted_out"
    else:
        # Anonymization step — THE one anonymize chain (A), shared with
        # every other cloud-egress path.  The local model is the SOLE
        # scope authority: it classifies real values against ``scrub``
        # and returns BOTH the real_name -> placeholder mapping AND its
        # own rewrite of the transcript with those values placeholdered.
        # Phase trace captures the raw model JSON so calibration can diff
        # prompt variants on the anonymizer in isolation.
        with phase_trace("anonymize") as t:
            payload = anonymize_for_cloud(
                graph,
                model,
                tokenizer,
                transcript=transcript,
                scrub=scrub,
                speaker_name=speaker_name,
                max_tokens=max_tokens,
                seed=seed,
                prompts_dir=prompts_dir,
            )
            t.set_raw(payload.raw)
            t.set_parsed(
                {
                    "mapping": dict(payload.forward),
                    "mapping_size": len(payload.forward),
                    "parse_ok": payload.status != "failed",
                    "anonymized_transcript_len": len(payload.anon_transcript or ""),
                }
            )
            if payload.status == "failed":
                t.set_outcome(
                    "failed",
                    reason="anonymization parse failed or missing/empty anonymized_transcript",
                )
            elif not graph.relations:
                t.set_outcome("no_input", reason="graph has 0 relations")
        _vram_snapshot(f"after_anonymize session={graph.session_id}")
        if payload.status == "failed":
            # Fail-closed: parse failure OR a missing/empty
            # anonymized_transcript.  Never fall back to raw plausibility
            # on the ORIGINAL real-name transcript — fall back to local
            # plausibility on the LOCAL-EXTRACT facts instead (no cloud
            # egress at all).
            logger.warning("Anonymization failed — falling back to raw plausibility")
            graph.diagnostics["anonymize"] = "failed"
            return StageState(
                graph=_fallback_plausibility_on_raw(
                    graph,
                    transcript,
                    model,
                    tokenizer,
                    "anon_failed",
                    speaker_name=speaker_name,
                    speaker_id=speaker_id,
                    max_tokens=max_tokens,
                    plausibility_max_tokens=plausibility_max_tokens,
                    seed=seed,
                ),
                original_relation_count=original_count,
            )
        if chain_stopped():
            # Calibration short-circuit: anonymize completed; downstream
            # phases (sota_enrich, anon_plausibility, deanon, …) are skipped.  graph.relations
            # remains the local-extract output; the anonymize result lives in
            # graph.diagnostics["phases"][anonymize].parsed.
            return StageState(graph=graph, original_relation_count=original_count)
        # ``payload.norm_stats`` is the LIVE signal — reaches
        # ``mapping_ambiguous_dropped`` unconditionally now (this is the
        # only normalize call in the chain; see cloud_egress.py).
        if payload.norm_stats["dropped"]:
            graph.diagnostics["mapping_ambiguous_dropped"] = payload.norm_stats["dropped"]

        # CORE-map diagnostic.  The CORE map is never otherwise
        # persisted — ``anonymize.parsed.mapping`` above is the LLM HINT map,
        # recorded before ``anonymize_for_cloud``'s table build runs, not
        # CORE.  Keys and COUNTS only — never the real names:
        # ``graph_snapshot.json`` (debug dumps under ``data/ha/debug/``)
        # serializes ``graph.diagnostics`` wholesale, and the placeholder
        # keyset is non-identifying while the values are the PII.
        graph.diagnostics["core_placeholders"] = {
            "keys": sorted(payload.reverse.keys()),
            "count": len(payload.reverse),
        }
        graph.diagnostics["anonymize"] = "ok"

    # ``reverse_mapping`` / ``anon_transcript`` are the chain's outputs
    # regardless of which branch above ran — THE speaker-value guard in
    # ``_build_anonymization_mapping`` (paramem/graph/placeholders.py) now
    # applies on every path by construction (payload.reverse is produced
    # exclusively by that function inside ``anonymize_for_cloud``).
    reverse_mapping = payload.reverse
    anon_transcript = payload.anon_transcript

    # Phase — entity_correction.  Local model classifies+corrects misspelled
    # real place/organization/concept surfaces at two loci: the reverse map
    # VALUES ONLY (not keys — the forward map used to anonymize the
    # transcript below, and every downstream identity check keyed on
    # placeholders, are unaffected) and, when the "attributes" knob member
    # is set, graph.entities[*].attributes values (e.g. current_location).
    # correct_entity_surfaces() is read-only over its inputs — it returns
    # accepted corrections as data, applied below onto reverse_mapping and
    # graph.entities (the same graph object this function returns, so the
    # applied change reaches keyed-entry assembly/indexed-key distillation
    # downstream). Correction is independent of cloud enrichment and safely
    # precedes the fact-construction block below (placeholder keys are
    # untouched). See paramem.graph.entity_correction for the full contract.
    with phase_trace("entity_correction") as t:
        correction_result = correct_entity_surfaces(
            reverse_mapping,
            graph.entities,
            model,
            tokenizer,
            correction_entity_types=correction_entity_types,
            prompts_dir=prompts_dir,
            model_alias=model_alias,
            seed=seed,
        )
        applied = correction_result["applied"]
        verdicts = correction_result["verdicts"]
        # correct_entity_surfaces is read-only over its inputs — it returns
        # accepted corrections as data; apply them here so the mutation
        # reaches reverse_mapping (used below for anonymized-transcript
        # substitution) and graph.entities (feeds keyed-entry assembly).
        for entry in applied:
            if entry["locus"] == "placeholder":
                reverse_mapping[entry["placeholder"]] = entry["after"]
            elif entry["locus"] == "attribute":
                graph.entities[entry["entity_index"]].attributes[entry["key"]] = entry["after"]
        if applied:
            graph.diagnostics["entity_corrections"] = applied
        graph.diagnostics["entity_correction_verdicts"] = verdicts
        t.set_parsed(
            {"applied_count": len(applied), "rejected_count": len(verdicts) - len(applied)}
        )
    if chain_stopped():
        # Calibration short-circuit: entity_correction completed; downstream
        # phases (sota_enrich, anon_plausibility, deanon, …) are skipped.  graph.relations
        # remains the local-extract output; the correction result lives in
        # graph.diagnostics["entity_corrections"] (applied only) /
        # graph.diagnostics["entity_correction_verdicts"] (every evaluated
        # target) / phases[entity_correction].
        return StageState(graph=graph, original_relation_count=original_count)

    # The fact array — ``payload.anon_facts``, built inside
    # ``anonymize_for_cloud`` (THE one construction, shared with the graph
    # and chat-egress tiers — not reimplemented).  Facts are never taken
    # from the model's response — the model's job is the TRANSCRIPT
    # (``anon_transcript``, already built above); the fact array is
    # always deterministic.  A fact can therefore never be lost, reworded,
    # or dropped by the anonymizer, and a placeholder cannot be glued into
    # a predicate at this stage — the motivating bug
    # (``language_proficiency_Language_3``) cannot occur here.  It can
    # still occur in SOTA's *returned* facts, which is why the
    # deanon-stage predicate invariant (:func:`_apply_bindings`) stays.
    # An orphan placeholder in a fact is likewise impossible: every
    # placeholder a fact can carry comes from this same forward map.
    # Correct to reuse here even though ``entity_correction`` ran between
    # this payload's construction and this line — correction mutates only
    # the REVERSE map's values, never the forward map ``anon_facts`` was
    # built from.  With an empty ``mapping`` (opt-out), substitution is a
    # no-op and facts egress verbatim.
    anon_facts = payload.anon_facts

    # Phase — sota_enrich.  Cloud (Anthropic by default) runs the
    # enrichment prompt; emits enriched facts + new_entity_bindings +
    # updated_anon_transcript.
    #
    # ``observed`` (computed inside ``CloudScope.for_response`` below) =
    # every placeholder token SOTA is actually shown — the rendered
    # facts_json (subject/predicate/object; a placeholder in a predicate
    # is still visible to SOTA) and the anonymized transcript.  This is
    # CORE's legality domain for this SOTA cycle: only tokens SOTA
    # actually saw may be treated as legitimately bound.
    with phase_trace("sota_enrich") as t:
        # :func:`_sota_facing_payload` is the SAME render
        # :func:`_filter_with_sota` uses for its prompt, so the two cannot
        # drift.
        _facts_text, _transcript_text = _sota_facing_payload(anon_facts, anon_transcript)
        # Send anon facts and transcript to SOTA as the SCRIPT built them
        # (the anonymizer LLM returns the mapping and its own anonymized
        # transcript, but it never produces facts). The SOTA prompt's convention
        # is "anonymizer placeholders are bare; only new entities
        # introduced by SOTA use braced form (`{Prefix_N}`)". SOTA also
        # returns explicit bindings for any braced placeholders it
        # minted, so de-anonymization is pure dict substitution
        # downstream — no transcript diff, no LLM call, no regex
        # post-processing.
        (
            enriched_anon,
            updated_anon_transcript,
            sota_bindings,
            _sota_raw,
            _sota_info,
        ) = _filter_with_sota(
            anon_facts,
            api_key,
            provider,
            filter_model,
            anon_transcript,
            endpoint=endpoint,
            max_tokens=max_tokens,
            prompts_dir=prompts_dir,
        )
        t.set_raw(_sota_raw or "")
        if _sota_info:
            graph.diagnostics["sota_call_info"] = _sota_info
            t.add("sota_call_info", _sota_info)
        if enriched_anon is None:
            # FAIL the cycle.  Previously fell back to anon_facts, which
            # silently baked a degraded (un-enriched) snapshot into the
            # cumulative graph — the same triples re-extracted in the
            # next cycle would dedup, so the missing second-order
            # relations were lost permanently.  Extraction failure must
            # fail the whole cycle: raise and propagate
            # past :meth:`ConsolidationLoop.extract_session` (which has
            # not yet merged this session's graph), and let the
            # per-session loop in app.py treat this session like a
            # ``VramExhausted`` chunk — leave it pending and retry on
            # the next cycle.
            t.set_parsed(
                {
                    "input_count": len(anon_facts),
                    "output_count": 0,
                    "new_bindings_count": 0,
                    "new_bindings": {},
                    "updated_anon_transcript_len": 0,
                }
            )
            t.set_outcome("failed", reason="SOTA call failed or unparseable")
            raise ExtractionFailed(
                "sota_enrich",
                "cloud enrichment call failed or response unparseable",
            )
        # Binding-totality gate — the ONE anonymize/deanonymize round-trip
        # scope for this response.  ``CloudScope.for_response`` computes
        # ``observed`` from the DECLARED vocabulary and the rendered
        # payload (never a shape scrape — see its docstring); a non-empty
        # verdict means an orphan mint or a CORE/SOTA conflict, and the
        # delta is REJECTED AS A WHOLE (not partially applied) so a bad
        # mint can never shed the local facts its ``drop`` action
        # replaced.  ``deanonymize_facts`` runs the gate unconditionally
        # as step 1 — it cannot be skipped from this call site.  This
        # call's ``.facts`` is intentionally NOT the final substituted
        # output: the anon-stage plausibility filter below may still
        # shrink ``enriched_anon`` before the "deanon" phase performs the
        # actual substitution (mirroring the pre-unification two-call
        # structure — check here, substitute later, exactly as
        # ``_check_mapping_totality`` then ``_apply_bindings`` used to be
        # two separate call sites).
        scope = CloudScope.for_response(
            payload, sota_bindings=sota_bindings, sent=(_facts_text, _transcript_text)
        )
        gate = deanonymize_facts(scope, enriched_anon)
        _record_binding_diagnostics(graph, gate)
        if gate.verdict:
            discarded_count = len(enriched_anon)
            retained_count = len(anon_facts)
            logger.error(
                "SOTA enrichment binding-totality breach: %d offending token(s) %s — "
                "rejecting the whole delta (%d enriched fact(s) discarded), falling "
                "back to %d local-extract fact(s).",
                len(gate.verdict),
                gate.verdict[:5],
                discarded_count,
                retained_count,
            )
            # The rejection: exactly three assignments.  ``anon_facts``
            # (the local-extract facts) is what saves the data — the
            # enrichment delta never touches it.  A rejected delta must be
            # indistinguishable downstream from a no-op delta; these three
            # assignments achieve exactly that.
            enriched_anon = anon_facts
            sota_bindings = {}
            updated_anon_transcript = None
            graph.diagnostics["sota_enrichment_rejected"] = gate.verdict
            t.set_outcome("rejected", reason=f"binding-totality breach: {gate.verdict[:5]}")
            t.set_parsed(
                {
                    "input_count": len(anon_facts),
                    "output_count": len(enriched_anon),
                    "new_bindings_count": 0,
                    "new_bindings": {},
                    "updated_anon_transcript_len": 0,
                    "observed_count": len(scope.observed),
                    "mapped_count": len(scope.core_resolution),
                }
            )
            # The scope for the "deanon" phase below must reflect the
            # reset ``sota_bindings`` — never apply a rejected mint.
            scope = CloudScope.for_response(
                payload, sota_bindings={}, sent=(_facts_text, _transcript_text)
            )
        else:
            t.set_parsed(
                {
                    "input_count": len(anon_facts),
                    "output_count": len(enriched_anon),
                    "new_bindings_count": len(sota_bindings or {}),
                    "new_bindings": dict(sota_bindings) if sota_bindings else {},
                    "updated_anon_transcript_len": len(updated_anon_transcript or ""),
                    "observed_count": len(scope.observed),
                    "mapped_count": len(scope.resolution),
                }
            )
            if not enriched_anon:
                logger.info("SOTA enrichment removed all relations")
                empty_cause = CAUSE_CLOUD_EMPTY
    if chain_stopped():
        # Calibration short-circuit: SOTA enrichment block recorded,
        # downstream (anon_plausibility, deanon, deanon_plausibility) skipped.
        # graph.relations stays at the local-extract output; enrichment result
        # is in phases[sota_enrich].
        return StageState(graph=graph, original_relation_count=original_count)

    # Step 3a: Plausibility on anonymized data (SOTA judge, stage="anon").
    # Only runs when: explicit SOTA provider, plausibility_stage=="anon",
    # and enriched_anon is non-empty.
    # Guard: use `plausibility_judge in PROVIDER_KEY_ENV` (NOT != "off") —
    # "auto" is not a provider and would crash PROVIDER_KEY_ENV.get("auto").
    if plausibility_stage == "anon" and plausibility_judge in PROVIDER_KEY_ENV and enriched_anon:
        with phase_trace("anon_plausibility") as t:
            judge_verdict = evaluate_cloud_egress(
                sota_enabled=True,
                provider=plausibility_judge,
                model=plausibility_model,
                endpoint=plausibility_endpoint,
            )
            if not judge_verdict.permitted:
                reason = "; ".join(judge_verdict.gaps)
                t.set_outcome("skipped", reason=reason)
                logger.warning(
                    "Anon-stage plausibility (%s) skipped — %s", plausibility_judge, reason
                )
            else:
                plaus_facts, plaus_raw = _plausibility_filter_with_sota(
                    enriched_anon,
                    judge_verdict.api_key,
                    provider=judge_verdict.provider,
                    filter_model=judge_verdict.model,
                    anon_transcript=updated_anon_transcript or anon_transcript,
                    endpoint=judge_verdict.endpoint,
                    max_tokens=max_tokens,
                    temperature=_DEFAULT_FILTER_TEMPERATURE,
                    prompts_dir=prompts_dir,
                )
                # Cloud round-trip can take 30–90s during which the WSL2 GPU
                # goes idle and the next local CUDA op fails with
                # "device not ready". Wake + settle before the deanon-stage
                # local plausibility filter that follows below.
                _wait_for_gpu_ready()
                t.set_raw(plaus_raw or "")
                if plaus_facts is not None:
                    pre_plaus = len(enriched_anon)
                    enriched_anon = plaus_facts
                    dropped_plaus = pre_plaus - len(enriched_anon)
                    graph.diagnostics["plausibility"] = "anon"
                    # Own key: the three plausibility writers (this one,
                    # the deanon judge, and the raw-fallback judge) used
                    # to share ``plausibility_dropped`` with three
                    # different semantics — a plain overwrite here, an
                    # accumulate there — so its final value was
                    # order-dependent and uninterpretable.
                    graph.diagnostics["plausibility_dropped_anon"] = dropped_plaus
                    graph.diagnostics["plausibility_judge_actual"] = plausibility_judge
                    if not enriched_anon:
                        empty_cause = CAUSE_ANON_JUDGE
                    if plaus_raw:
                        graph.diagnostics["sota_plausibility_raw_response"] = plaus_raw
                    t.set_parsed(
                        {
                            "judge": plausibility_judge,
                            "input_count": pre_plaus,
                            "kept_count": len(enriched_anon),
                            "dropped_count": dropped_plaus,
                        }
                    )
                    logger.info(
                        "Anon-stage plausibility (%s): %d → %d facts (%d dropped)",
                        plausibility_judge,
                        pre_plaus,
                        len(enriched_anon),
                        dropped_plaus,
                    )
                else:
                    t.set_outcome("failed", reason="plausibility call returned None")
                    t.set_parsed(
                        {
                            "judge": plausibility_judge,
                            "input_count": len(enriched_anon),
                            "kept_count": len(enriched_anon),
                            "dropped_count": 0,
                        }
                    )
                    logger.warning("Anon-stage plausibility call failed — keeping enriched facts")
        if chain_stopped():
            # Calibration short-circuit after the optional anon-stage judge.
            return StageState(graph=graph, original_relation_count=original_count)

    # Empty-check guard: if enriched_anon is empty after the anon-stage
    # judge (or was already empty), clear the graph and hand back a state
    # whose empty ``facts`` is the composite spec's ``terminal_when`` — the
    # deanonymize/rebuild siblings must not run.
    if not enriched_anon:
        logger.info("No facts remain after anon-stage plausibility — returning empty graph")
        graph.relations = []
        graph.entities = []
        return StageState(
            graph=graph,
            original_relation_count=original_count,
            empty_cause=empty_cause,
        )

    # Hand-over to the ``deanonymize`` sibling: de-anonymization via
    # state-machine substitution, the deanon-stage judge and the
    # relation/entity rebuild all live in their own stages now.  ``scope``
    # is the ONE anonymize/de-anonymize round-trip scope for this response
    # (rebuilt above when the enrichment delta was rejected, so a rejected
    # mint can never be applied); the substitution and the entity-type
    # rebuild are both keyed on it.
    return StageState(
        graph=graph,
        facts=enriched_anon,
        scope=scope,
        sota_raw=_sota_raw,
        updated_anon_transcript=updated_anon_transcript,
        original_relation_count=original_count,
        empty_cause=empty_cause,
    )


# _is_scalar_value / _partition_scalar_facts /
# _project_scalar_facts_to_attributes now live in
# paramem.graph.relation_build (imported above) — the scalar routing is
# owned by the ``rebuild`` stage; the ``deanonymize`` stage only invokes
# the partition, which must run before the deanon-stage judge.

# load_anonymization_prompt / anonymize_with_local_model now live in
# paramem.graph.cloud_egress (imported above) — the local-model
# anonymizer call is reached exclusively through
# cloud_egress.anonymize_for_cloud; nothing in this module calls it
# directly any more.

# The provider tables (PROVIDER_KEY_ENV, OPENAI_COMPAT_ENDPOINTS,
# OPENAI_COMPAT_PROVIDERS) and the key resolver now live in
# paramem.utils.cloud_admission (imported above), alongside
# evaluate_cloud_egress — the one place the "may we reach a cloud LLM,
# and with what credentials?" decision is computed.  They are imported
# here for dispatch (_sota_call / _filter_openai_compat), not owned here.


# The three SOTA system prompts below (``sota_enrichment_system.txt``,
# ``sota_plausibility_system.txt``, ``sota_graph_enrichment_system.txt``)
# are loaded at CALL TIME inside each consuming function (``_filter_with_sota``,
# ``_graph_enrich_with_sota``, ``_plausibility_filter_with_sota``,
# ``local_plausibility_filter``) — never as module-level constants.  A
# module-level ``_load_prompt(...)`` call runs at import time, before any
# :func:`~paramem.graph.phase_trace.extraction_trace`/``phase_trace`` scope
# can exist and before :func:`~paramem.graph.prompts.prompt_overrides` can
# be active, so it is permanently unreachable by both prompt provenance and
# calibration overrides.  ``_filter_anthropic`` / ``_filter_openai_compat`` /
# ``_sota_call`` still default their own ``system_prompt`` parameter to the
# enrichment prompt for callers that omit it (resolved lazily, in the body,
# never as a default-expression — a default expression is itself evaluated
# at function-definition time, i.e. import time, so it has the exact same
# unreachability problem as a module-level constant).


# _DEFAULT_FILTER_MAX_TOKENS / _DEFAULT_FILTER_TEMPERATURE /
# _DEFAULT_FILTER_TIMEOUT_SECONDS are defined at the top of this module —
# they need to precede the extract_graph signature that references them.


def _filter_anthropic(
    prompt: str,
    api_key: str,
    filter_model: str,
    system_prompt: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str | None:
    """Call Anthropic with a single user message; return raw text or ``None``
    on transport / SDK failure.

    ``top_p`` / ``top_k`` are optional sampling overrides used by the
    calibration tool to probe SOTA non-determinism.  Anthropic's API does
    not accept a ``seed`` parameter so seed-based reproducibility cannot
    be requested at this layer; the calibration tool reports
    ``params_effective.seed=null`` for SOTA stages so the operator knows
    it was dropped.  Both default to ``None`` — production paths
    preserve current temperature-only sampling behaviour.

    ``system_prompt`` defaults to ``None``, resolved here (never as a
    default-expression, which would evaluate at import time) to the
    ``sota_enrichment.txt`` system prompt — production callers
    (:func:`_sota_call`, forwarded from :func:`_filter_with_sota`) always
    pass their own resolved prompt explicitly; the fallback here only
    serves a direct caller (e.g. a test) that omits it.
    """
    if system_prompt is None:
        system_prompt = _load_prompt("sota_enrichment_system.txt", required=True)
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping SOTA filter")
        return None
    extra_kwargs: dict = {}
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if top_k is not None:
        extra_kwargs["top_k"] = top_k
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        response = client.messages.create(
            model=filter_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **extra_kwargs,
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        cause = e.__cause__ or e.__context__
        detail = f"{type(e).__name__}: {e}"
        if cause:
            detail += f" (caused by {type(cause).__name__}: {cause})"
        logger.warning("Anthropic API call failed — %s", detail)
        return None


def _filter_openai_compat(
    prompt: str,
    api_key: str,
    filter_model: str,
    provider: str,
    endpoint: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> str | None:
    """Call an OpenAI-compatible chat-completions endpoint; return raw
    text or ``None`` on transport / SDK failure.

    ``system_prompt`` defaults to ``None``, resolved here (never as a
    default-expression, which would evaluate at import time) to the
    ``sota_enrichment.txt`` system prompt — see :func:`_filter_anthropic`
    for why.
    """
    if system_prompt is None:
        system_prompt = _load_prompt("sota_enrichment_system.txt", required=True)
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — skipping SOTA filter")
        return None

    url = endpoint or OPENAI_COMPAT_ENDPOINTS.get(provider)
    if not url:
        logger.warning("No endpoint for OpenAI-compatible provider '%s'", provider)
        return None
    payload = {
        "model": filter_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, httpx.RequestError, KeyError, IndexError, ValueError, TypeError) as e:
        # ValueError covers resp.json() raising json.JSONDecodeError on a
        # 200 response with a non-JSON body (a proxy error page, a captive-
        # portal interstitial, a truncated stream — real behind a VPN);
        # json.JSONDecodeError is a ValueError subclass, so it is caught by
        # name here rather than imported separately.  TypeError covers an
        # unexpected JSON shape (e.g. "choices" is null).  Both are boundary
        # conditions of an external API returning garbage, not programming
        # errors — same failure contract as _filter_anthropic's broad catch,
        # so callers (e.g. the graph-tier chunk loop in consolidation.py,
        # narrowed to except RuntimeError) see a clean None instead of an
        # escaping exception that kills the whole fold over one chunk.
        logger.warning("%s API call failed: %s", provider, e)
        return None


def _sota_call(
    prompt: str,
    api_key: str,
    provider: str,
    filter_model: str,
    endpoint: str | None,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> str | None:
    """Generic SOTA dispatch (anthropic native or any OpenAI-compatible host).

    ``system_prompt`` defaults to ``None`` and is forwarded as-is to
    ``_filter_anthropic``/``_filter_openai_compat`` — each resolves its own
    ``None`` there (never here as a default-expression, which would
    evaluate at import time); production callers
    (:func:`_filter_with_sota`, :func:`_graph_enrich_with_sota`,
    :func:`_plausibility_filter_with_sota`) always pass their own resolved
    prompt explicitly.
    """
    if provider == "anthropic":
        return _filter_anthropic(
            prompt,
            api_key,
            filter_model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    if provider in OPENAI_COMPAT_PROVIDERS:
        return _filter_openai_compat(
            prompt,
            api_key,
            filter_model,
            provider,
            endpoint,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    logger.warning("Unsupported SOTA provider '%s'", provider)
    return None


def _parse_facts_response(raw: str | None, strict_array: bool = False) -> list[dict] | None:
    """Parse a SOTA response into a list of fact dicts. Returns None on failure.

    `strict_array=True` rejects dict-wrapped responses — used by the
    plausibility filter, whose contract requires a bare JSON array. The
    enrichment stage is more permissive (tries common dict keys before failing).

    When the strict envelope parse fails (typically because Mistral 7B emits
    EOS mid-array on long KEEP-by-default plausibility passes — the closing
    ``]`` never arrives), a stream-parse salvage walks ``{…}`` objects from
    the response and returns those that look fact-shaped.  Each salvaged
    object must carry at least one of ``subject`` / ``predicate`` / ``object``
    so unrelated JSON inside the response (preamble, commentary literals)
    isn't pulled into the result.
    """
    if raw is None:
        return None
    logger.debug("SOTA response raw: %s", raw[:500])
    try:
        json_str = _extract_json_block(raw)
        validated = json.loads(json_str)
        if isinstance(validated, list):
            return validated
        if not strict_array and isinstance(validated, dict):
            for key in ("relations", "filtered", "facts", "results"):
                if key in validated and isinstance(validated[key], list):
                    return validated[key]
        logger.warning("SOTA response unexpected format: %s", type(validated).__name__)
        return None
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.debug("SOTA response strict parse failed: %s — attempting salvage", e)
        salvaged = _salvage_fact_objects(raw)
        if salvaged:
            logger.warning(
                "SOTA response strict parse failed (%s); salvaged %d fact dict(s) "
                "via stream-parse — likely a truncated array",
                e,
                len(salvaged),
            )
            return salvaged
        logger.warning("SOTA response parse failed: %s", e)
        return None


def _salvage_fact_objects(raw: str) -> list[dict]:
    """Stream-parse ``{…}`` fact objects from a malformed JSON envelope.

    Walks the response and yields each balanced ``{…}`` block.  Each block
    is parsed with ``json.loads``; successful parses that look fact-shaped
    (carry ``subject``, ``predicate``, or ``object``) are kept.  Used as a
    fallback when the envelope is truncated mid-array (no closing ``]``)
    so the strict parse can't recover anything.

    Conservative on inclusion: an object with none of the fact keys is
    dropped to avoid pulling commentary literals (``{"note": "..."}``) or
    the SOTA-style ``new_entity_bindings`` sub-dict into a fact list.
    Returns an empty list when no fact-shaped objects can be recovered.
    """
    if not raw:
        return []
    salvaged: list[dict] = []
    depth = 0
    in_string = False
    escape = False
    start: int | None = None
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                # Stray closer — ignore.
                continue
            depth -= 1
            if depth == 0 and start is not None:
                block = raw[start : i + 1]
                start = None
                try:
                    obj = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if any(k in obj for k in ("subject", "predicate", "object")):
                    salvaged.append(obj)
    return salvaged


def _filter_with_sota(
    anon_facts: list[dict],
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    anon_transcript: str | None = None,
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "sota_enrichment.txt",
) -> tuple[list[dict] | None, str | None, dict[str, str], str | None, dict]:
    """SOTA enrichment pass — coreference + compound splitting + safe reification.

    Returns ``(facts, updated_transcript, bindings, raw_response, info)``.

    The SOTA emits a delta envelope ``{"add": [...], "modify": [...],
    "drop": [...], "bindings": {...}}`` describing what to change against
    the indexed input facts. KEEP is the default; unnamed input facts pass
    through unchanged. The transcript is rendered locally from
    ``anon_transcript`` plus ``bindings`` — never carried back on the
    wire — so output bandwidth is bounded by the size of the change set,
    not by the size of the input.

    ``bindings`` maps each new braced placeholder SOTA introduced (key
    without braces, e.g. ``"Event_1"``) to the exact transcript span it
    stands for. SOTA already knows the binding the moment it mints each
    placeholder, so emitting it explicitly removes the transcript-diff
    reconstruction step the previous "echo every fact" protocol relied on.

    ``info`` is a dict with diagnostic flags the caller persists into
    ``graph.diagnostics``:

    * ``parse_path``: ``"delta"`` (envelope parsed, delta applied),
      ``"failed"`` (parse failure), or ``"no_response"`` (provider
      returned nothing). Both failure paths return ``None`` facts, and
      the caller (:func:`_sota_pipeline`) raises
      :class:`ExtractionFailed` on either — it does NOT fail open.
      Failing open used to silently bake a degraded, un-enriched
      snapshot into the cumulative graph, where the next cycle's
      re-extraction deduped it and the missing relations were lost
      permanently; see the call site for the full reasoning.
    * ``response_chars``: length of the raw response in characters.
    * ``add_count`` / ``modify_count`` / ``drop_count``: validated
      action counts; entries that fail per-entry validation (out-of-range
      indices, non-dict fields) are not counted.
    * ``bindings_count``: number of SOTA-introduced placeholders for
      which the response carried an explicit binding.

    The prompt this function loads is external config — edit
    ``configs/prompts/sota_enrichment.txt`` to tune; no code changes are
    needed.

    ``prompts_dir`` overrides the search directory (forwarded to
    :func:`_load_prompt`) so a calibration override actually reaches the
    model; it defaults to the production template. ``prompt_filename``
    overrides the file name within that directory (the production caller,
    :func:`_sota_pipeline`, keeps the default); consistent with the
    ``prompt_filename`` parameter on :func:`anonymize_with_local_model` and
    :func:`local_plausibility_filter`.
    """
    enrichment_prompt = _load_prompt(prompt_filename, prompts_dir=prompts_dir, required=True)
    system_prompt = _load_prompt("sota_enrichment_system.txt", required=True)
    facts_json, transcript_text = _sota_facing_payload(anon_facts, anon_transcript)
    prompt = enrichment_prompt.format(facts_json=facts_json, transcript=transcript_text)
    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
    )
    if raw is None:
        return None, None, {}, None, {"parse_path": "no_response"}
    surviving, updated_transcript, bindings, counts = _apply_enrichment_delta(
        anon_facts, raw, anon_transcript
    )
    info: dict = {"response_chars": len(raw), **counts}
    if surviving is None:
        info["parse_path"] = "failed"
        logger.warning(
            "SOTA enrichment delta parse failed (response_chars=%d) — "
            "caller will fail the extraction (ExtractionFailed)",
            info["response_chars"],
        )
    else:
        info["parse_path"] = "delta"
    return surviving, updated_transcript, bindings, raw, info


# ---------------------------------------------------------------------------
# Graph-level SOTA enrichment (Task #10)
# ---------------------------------------------------------------------------


def _graph_enrich_with_sota(
    triples: list[dict],
    payload: AnonymizedPayload,
    graph: SessionGraph,
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> tuple[list[dict], list[list[str]], str | None, list[str]] | None:
    """SOTA graph-level enrichment pass over a pre-merged cumulative graph.

    Sends a subgraph serialized as triples to a SOTA provider and requests
    two outputs:
    - New cross-session second-order relations not already in the graph.
    - ``same_as`` pairs identifying duplicate nodes under different surface forms.

    Runs the SAME anonymize -> SOTA -> de-anonymize contract as the
    session-tier :func:`_sota_pipeline`, via the SAME chain in
    :mod:`paramem.graph.cloud_egress` — this is the second call site of
    that contract. ``payload`` is (A)'s output for this chunk — the
    caller (:func:`~paramem.training.graph_enrich.run_graph_enrichment`)
    already ran
    :func:`~paramem.graph.cloud_egress.anonymize_for_cloud` (with
    ``identity_domain`` reconciliation and the domain-scoped fail-closed
    guard) before calling this function; this function applies no scope
    gate of its own — it only substitutes and de-anonymizes.

    ``payload.reverse`` is produced exclusively by
    :func:`~paramem.graph.placeholders._build_anonymization_mapping`
    inside (A) — the speaker-value guard applies here by construction; no
    code path in this function inverts an unfiltered forward map.

    The chunk's anonymized ``subject``/``object`` fields come from
    ``payload.anon_facts`` (built by (A) step 8, in the SAME order as
    ``triples`` — the chunk graph consolidation.py builds is derived from
    ``triples`` in order); ``predicate``, ``relation_type``, and
    ``speaker_id`` are copied VERBATIM from ``triples`` — a speaker id can
    never be a mapping key (the anonymization prompt instructs the model
    to leave ``speaker{N}`` ids verbatim), so no substitution was ever
    needed there.

    After the SOTA call, the response's ``relations`` are de-anonymized via
    :func:`~paramem.graph.cloud_egress.deanonymize_facts` (the single exit
    gate: totality gate, THEN predicate invariant, substitute, residual
    sweep, fail-closed) — returning real node names and bare ``speaker{N}``
    ids, exactly what
    :func:`~paramem.training.graph_enrich.run_graph_enrichment`'s existing
    consumption logic (including the speaker-pair guard that rejects a same_as pair
    where both surfaces are speaker ids) already expects. A non-empty
    totality verdict REJECTS THE WHOLE CHUNK DELTA (relations AND
    ``same_as`` both discarded) — the caller detects this from the
    verdict this function RETURNS (the fourth tuple element) and counts it
    in ``totality_rejected_chunks``.
    Any ``bindings`` the response carries (SOTA-minted placeholders —
    normally empty, since the prompt forbids inventing new nodes) are
    normalized inside :meth:`~paramem.graph.cloud_egress.CloudScope.
    for_response` (placeholder on the key side) before being folded into
    that substitution, mirroring the session tier. ``same_as`` pairs are
    restored per-member via
    :func:`~paramem.graph.cloud_egress.deanonymize_response_text` — a pair
    with either member unresolved (``None``) is dropped.

    ``deanonymize_response_text`` deliberately does NOT run the
    undeclared-orphan SHAPE backstop (:data:`~paramem.graph.placeholders.
    PLACEHOLDER_TOKEN_RE`) that :func:`~paramem.graph.placeholders.
    _apply_bindings` runs for fact fields — see that function's docstring
    for why (free conversational prose is a wide false-positive surface
    for a bare shape match, e.g. ``GPT_4``, ``COVID_19``). That rationale
    does NOT apply to a ``same_as`` member — it is a structured node-name
    field, not prose — but the omission is still safe HERE specifically,
    for a different reason: an undeclared placeholder-shaped token that
    survives ``deanonymize_response_text`` unresolved (e.g. a stray
    ``"Person_99"`` SOTA invented) is never itself a real node in this
    process's graph, so
    :func:`~paramem.training.graph_enrich.run_graph_enrichment`'s own
    downstream guard (``keep_canon not in graph or drop_canon not in
    graph`` — every ``same_as`` member must resolve to an EXISTING graph
    node) drops the pair unconditionally before any contraction happens.
    This is a real, load-bearing safety property of THIS call site, not a
    silently-absorbed gap — if a future consumer of ``same_as`` pairs
    ever accepts an unknown node name (e.g. to synthesize one), this
    argument no longer holds and the shape backstop must be restored
    here explicitly.

    Under the production default (``scrub`` includes ``"person name"``, so
    the model's ``mapping`` is dominated by person nodes), person nodes —
    including speakers — are tokenised; org/place/thing nodes the model
    left out of ``mapping`` pass through verbatim. Accepted consequence:
    person-level ``same_as`` coreference (e.g. ``["Yang Ming", "Mr.
    Yang"]``) can no longer be detected by the cloud judge, since both
    surfaces collapse to opaque, unrelated tokens before the model ever
    sees them — the name-surface signal coreference depends on is gone for
    people. org/place/thing ``same_as`` is unaffected (those surfaces stay
    verbatim under the default ``scrub``).

    Loads ``sota_graph_enrichment.txt`` (required). The prompt uses a
    ``{triples_json}`` placeholder.

    Args:
        triples: List of ``{"subject", "predicate", "object", "relation_type",
            "speaker_id"}`` dicts representing the chunk subgraph, from
            :func:`~paramem.training.graph_enrich.serialize_subgraph_triples`
            (unchanged — anonymization happens on its output here, not
            inside it).
        payload: :class:`~paramem.graph.cloud_egress.AnonymizedPayload` —
            the caller's already-completed (A) result for this chunk.
        graph: The caller's throwaway per-chunk ``SessionGraph`` — the
            diagnostics sink this function writes the totality gate's
            findings to (via :func:`_record_binding_diagnostics`).  It is
            NOT how the caller learns of a rejection: that arrives as the
            returned verdict.
        api_key: Provider API key.
        provider: SOTA provider name (e.g. ``"anthropic"``).
        filter_model: Model identifier for the provider.
        endpoint: Custom endpoint for OpenAI-compatible providers.
        max_tokens: Maximum tokens in the SOTA response.
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        ``(new_relations, same_as_pairs, raw_response, totality_verdict)``
        on success, or ``None`` when the SOTA call fails or the response
        cannot be parsed.  ``new_relations`` is a list of relation dicts
        with real node names; ``same_as_pairs`` is a list of
        ``[canonical, variant]`` pairs with real node names / bare speaker
        ids.  ``totality_verdict`` is the binding-totality gate's verdict —
        ``[]`` on an accepted delta, the sorted offending tokens on a
        rejected one.  A rejected totality verdict is NOT ``None`` — it is
        ``([], [], raw_response, verdict)``; the non-empty verdict is what
        tells the caller "delta discarded" apart from "delta legitimately
        empty" (``([], [], raw_response, [])``).

    The prompt this function loads is external config — edit
    ``configs/prompts/sota_graph_enrichment.txt`` to tune; no code changes
    are needed.
    """
    anon_triples = [
        {**t, "subject": f["subject"], "object": f["object"]}
        for t, f in zip(triples, payload.anon_facts, strict=True)
    ]

    enrichment_prompt = _load_prompt("sota_graph_enrichment.txt", required=True)
    system_prompt = _load_prompt("sota_graph_enrichment_system.txt", required=True)
    # No try/except: a KeyError here means the prompt template has an
    # un-doubled literal brace (a template bug, not a runtime condition).
    # Swallowing it turned a missed brace-doubling into a permanent,
    # silent outage of graph enrichment — it must kill the fold loudly.
    triples_json = json.dumps(anon_triples, indent=2)
    prompt = enrichment_prompt.format(triples_json=triples_json)

    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
    )
    if raw is None:
        return None

    # Parse response: preferred schema {"relations": [...], "same_as": [...],
    # "bindings": {...}}; legacy bare-array (relations only).
    try:
        json_str = _extract_json_block(raw)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Graph enrichment response parse failed: %s", exc)
        return None

    if isinstance(parsed, list):
        logger.debug("Graph enrichment: bare-array response (no same_as)")
        new_relations: list = parsed
        raw_same_as: list = []
        raw_bindings: dict = {}
    elif isinstance(parsed, dict):
        new_relations = parsed.get("relations") or []
        raw_same_as = parsed.get("same_as") or []
        raw_bindings = parsed.get("bindings") or {}
        if not isinstance(new_relations, list):
            logger.warning("Graph enrichment: 'relations' is not a list, ignoring")
            new_relations = []
        if not isinstance(raw_bindings, dict):
            raw_bindings = {}
    else:
        logger.warning("Graph enrichment: unexpected response type %s", type(parsed).__name__)
        return None

    # Validate same_as entries: must be 2-element lists/tuples of non-empty strings.
    same_as_pairs: list[list[str]] = []
    for pair in raw_same_as:
        if (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
            and pair[0]
            and pair[1]
        ):
            same_as_pairs.append([pair[0], pair[1]])
        else:
            logger.debug("Graph enrichment: malformed same_as entry skipped: %r", pair)

    # THE one scope for this response — observed derived from the EXACT
    # triples_json string sent to SOTA (never a shape scrape).
    scope = CloudScope.for_response(payload, sota_bindings=raw_bindings, sent=(triples_json,))

    # De-anonymize relations — the SAME exit gate the session tier uses,
    # gated by the totality check as step 1 (unconditional — the graph
    # tier's binding-totality gate).  A non-empty verdict rejects the
    # WHOLE chunk delta.
    deanon = deanonymize_facts(scope, new_relations)
    _record_binding_diagnostics(graph, deanon)
    if deanon.verdict:
        logger.warning(
            "graph_enrichment: binding-totality breach: %d offending token(s) %s — "
            "rejecting the whole chunk delta (%d relation(s) discarded).",
            len(deanon.verdict),
            deanon.verdict[:5],
            len(new_relations),
        )
        return [], [], raw, deanon.verdict
    if deanon.predicate_dropped or deanon.residual_dropped:
        logger.warning(
            "graph_enrichment: dropped %d relation(s) post-substitution "
            "(%d predicate-invariant, %d residual placeholder sweep).",
            len(deanon.predicate_dropped) + len(deanon.residual_dropped),
            len(deanon.predicate_dropped),
            len(deanon.residual_dropped),
        )

    # same_as pairs: per-member free-text deanon — a pair with either
    # member unresolved (declared-but-unobserved, or otherwise fail-closed)
    # is dropped rather than forwarded with a residual placeholder.
    deanon_same_as: list[list[str]] = []
    for canon, variant in same_as_pairs:
        d_canon = deanonymize_response_text(scope, canon)
        d_variant = deanonymize_response_text(scope, variant)
        if d_canon is None or d_variant is None:
            logger.warning(
                "graph_enrichment: dropping same_as pair with unresolved token: %r",
                [canon, variant],
            )
            continue
        deanon_same_as.append([d_canon, d_variant])

    return deanon.facts, deanon_same_as, raw, deanon.verdict


def _render_indexed_facts(facts: list[dict]) -> str:
    """Format facts for the plausibility prompt as ``[N] <json>`` lines.

    The plausibility judge's output contract is a small ``{"drop": [...]}``
    object listing zero-based indices of facts that match a DROP rule.
    Rendering each input with its index in square brackets is what makes
    that contract referenceable — the judge can quote ``[3]`` rather than
    echoing the entire fact verbatim, which is what used to truncate
    Mistral 7B mid-array on long KEEP-by-default outputs.
    """
    return "\n".join(f"[{i}] {json.dumps(f, ensure_ascii=False)}" for i, f in enumerate(facts))


def _sota_facing_payload(facts: list[dict], anon_transcript: str | None) -> tuple[str, str]:
    """The exact ``(facts_json, transcript_text)`` pair rendered into
    every SOTA-facing prompt.

    ONE render so :func:`_filter_with_sota`'s enrichment prompt,
    :func:`_plausibility_filter_with_sota`'s prompt, and
    :func:`_sota_pipeline`'s ``observed`` legality-domain scan (the set
    of placeholder tokens SOTA was actually shown) cannot drift from one
    another — previously that invariant was enforced only by a code
    comment next to a hand-mirrored copy of the render.
    """
    return _render_indexed_facts(facts), anon_transcript or "(not available)"


def _parse_drop_set(raw: str | None, n_facts: int) -> set[int] | None:
    """Parse the plausibility judge's drop-set output.

    Accepts these shapes (most permissive — all are observed in practice):

    * ``{"drop": [0, 2, 5]}`` — the prompt's preferred shape.
    * ``[0, 2, 5]`` — bare integer array; some models drop the wrapper.
    * ``{"drop": [{"index": 0, "rule": "R1"}, ...]}`` — the model
      annotated each drop with its rule reason.  Indices are extracted;
      rules are ignored at parse time.

    Returns the drop set on success; ``None`` on parse failure (caller
    fail-opens — keep all facts).  Indices outside ``[0, n_facts)`` are
    skipped with a warning rather than failing the parse — a single bad
    index shouldn't void an otherwise-valid drop set.
    """
    if raw is None or not raw.strip():
        return None
    # Routes through the shared envelope finder (`_extract_json_block`).
    # That helper handles markdown fences, prose preamble before the
    # JSON, and the inline-backtick `{...}` wrapper implicitly (raw_decode
    # stops at the JSON's natural close; a trailing backtick is ignored).
    # Drop / drop_indices / indices are in `_JSON_ENVELOPE_KEYS`, and
    # bare integer arrays are accepted as envelopes for the same reason.
    try:
        json_str = _extract_json_block(raw)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("plaus drop-set parse failed: %s", e)
        return None
    candidates: list[object]
    if isinstance(parsed, dict):
        for key in ("drop", "drop_indices", "indices"):
            value = parsed.get(key)
            if isinstance(value, list):
                candidates = value
                break
        else:
            logger.warning(
                "plaus drop-set object missing 'drop' key (got keys: %s)",
                list(parsed.keys())[:5],
            )
            return None
    elif isinstance(parsed, list):
        candidates = parsed
    else:
        logger.warning("plaus drop-set unexpected shape: %s", type(parsed).__name__)
        return None
    drop: set[int] = set()
    out_of_range = 0
    for c in candidates:
        if isinstance(c, dict):
            idx = c.get("index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
        elif isinstance(c, bool) or not isinstance(c, int):
            continue
        else:
            idx = c
        if 0 <= idx < n_facts:
            drop.add(idx)
        else:
            out_of_range += 1
    if out_of_range:
        logger.warning(
            "plaus drop-set: %d index(es) out of range [0, %d) — skipped",
            out_of_range,
            n_facts,
        )
    return drop


def _apply_drop_set(facts: list[dict], raw: str | None) -> list[dict] | None:
    """Apply the judge's drop-set output to the input facts.

    Returns ``None`` on parse failure so the caller can fail-open
    (matches the prior contract: ``filtered_list is None`` →
    ``_sota_pipeline`` keeps all input facts unchanged and logs a
    warning).  Empty drop set → input list returned unchanged.
    """
    drop = _parse_drop_set(raw, len(facts))
    if drop is None:
        return None
    if not drop:
        return list(facts)
    return [f for i, f in enumerate(facts) if i not in drop]


def _parse_enrichment_delta(
    raw: str | None, n_facts: int
) -> tuple[list[dict], list[tuple[int, dict]], set[int], dict[str, str]] | None:
    """Parse the SOTA enrichment judge's delta-envelope output.

    Returns ``(add, modify, drop, bindings)`` on success; ``None`` on
    parse failure, which propagates through
    :func:`_apply_enrichment_delta` and :func:`_filter_with_sota` to a
    raised :class:`ExtractionFailed` at the ``sota_enrich`` call site —
    an unparseable enrichment response fails the cycle, it does not fall
    back to the pre-enrichment facts.

    * ``add``      — list of new fact dicts to append.
    * ``modify``   — list of ``(index, fields_dict)`` tuples; each entry
      is a partial update for the indexed input fact.
    * ``drop``     — set of zero-based indices to drop from the input.
    * ``bindings`` — dict mapping new ``"Prefix_N"`` placeholders to the
      exact anonymized-transcript spans they stand for.

    Tolerated shapes (mirroring ``_parse_drop_set``'s permissiveness):

    * ``{"add": [...], "modify": [...], "drop": [...], "bindings": {...}}``
      — preferred shape; all four keys optional (missing == no-op).
    * Markdown fences / prose preamble / inline-backtick wraps —
      unwrapped via the shared envelope finder.
    * ``null`` values for any key — coerced to empty.
    * ``new_entity_bindings`` accepted as a synonym of ``bindings`` so
      legacy-shape responses don't lose the binding payload silently.

    Indices outside ``[0, n_facts)`` in ``drop`` / ``modify`` are
    skipped with a warning rather than failing the whole parse — a
    single bad index shouldn't void an otherwise-valid delta.

    Every ``add`` entry and every ``modify`` entry's ``fields`` dict is
    restricted to :data:`_FACT_FIELDS` — the fields that actually reach
    ``Relation``.  A key outside that set (e.g. an ``evidence`` field an
    LLM invents) is stripped, not rejected: the fact itself (subject,
    predicate, object, ...) still applies, only the non-fact key is
    dropped, so it can never carry a residual placeholder into
    ``enriched_anon``, debug snapshots, or diagnostics — nor can it make
    the residual sweep in :func:`_apply_bindings` shed an otherwise-valid
    fact over a field that was never going to reach the graph anyway.
    """
    if raw is None or not raw.strip():
        return None
    try:
        json_str = _extract_json_block(raw)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("enrichment delta parse failed: %s", e)
        return None
    if not isinstance(parsed, dict):
        logger.warning(
            "enrichment delta unexpected shape: %s",
            type(parsed).__name__,
        )
        return None

    # add[] — every entry must be a dict; skip the rest. Restricted to
    # _FACT_FIELDS: any other key an LLM invents (e.g. "evidence") is
    # stripped here, before the entry ever enters enriched_anon.
    add: list[dict] = []
    unknown_fields_stripped = 0
    raw_add = parsed.get("add")
    if isinstance(raw_add, list):
        for entry in raw_add:
            if isinstance(entry, dict):
                unknown_fields_stripped += sum(1 for k in entry if k not in _FACT_FIELDS)
                add.append({k: v for k, v in entry.items() if k in _FACT_FIELDS})
    elif raw_add is not None:
        logger.warning(
            "enrichment delta: 'add' has non-list shape %s — ignored",
            type(raw_add).__name__,
        )

    # modify[] — each entry is {"index": <int>, "fields": {<partial>}};
    # tolerate either out-of-range indices or non-dict fields by skipping.
    # ``fields`` is restricted to _FACT_FIELDS for the same reason as
    # ``add`` above.
    modify: list[tuple[int, dict]] = []
    out_of_range_mod = 0
    raw_modify = parsed.get("modify")
    if isinstance(raw_modify, list):
        for entry in raw_modify:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            fields = entry.get("fields")
            if not isinstance(fields, dict):
                continue
            unknown_fields_stripped += sum(1 for k in fields if k not in _FACT_FIELDS)
            fields = {k: v for k, v in fields.items() if k in _FACT_FIELDS}
            if 0 <= idx < n_facts:
                modify.append((idx, fields))
            else:
                out_of_range_mod += 1
    elif raw_modify is not None:
        logger.warning(
            "enrichment delta: 'modify' has non-list shape %s — ignored",
            type(raw_modify).__name__,
        )
    if out_of_range_mod:
        logger.warning(
            "enrichment delta: %d modify index(es) out of range [0, %d) — skipped",
            out_of_range_mod,
            n_facts,
        )
    if unknown_fields_stripped:
        logger.warning(
            "enrichment delta: stripped %d non-fact field(s) outside %s from "
            "'add'/'modify' entries",
            unknown_fields_stripped,
            sorted(_FACT_FIELDS),
        )

    # drop[] — tolerates the same per-entry shapes as `_parse_drop_set`
    # (bare ints, `{"index": N, "rule": "Rk"}` annotated form).
    drop: set[int] = set()
    out_of_range_drop = 0
    raw_drop = parsed.get("drop")
    if isinstance(raw_drop, list):
        for c in raw_drop:
            if isinstance(c, dict):
                idx = c.get("index")
                if isinstance(idx, bool) or not isinstance(idx, int):
                    continue
            elif isinstance(c, bool) or not isinstance(c, int):
                continue
            else:
                idx = c
            if 0 <= idx < n_facts:
                drop.add(idx)
            else:
                out_of_range_drop += 1
    elif raw_drop is not None:
        logger.warning(
            "enrichment delta: 'drop' has non-list shape %s — ignored",
            type(raw_drop).__name__,
        )
    if out_of_range_drop:
        logger.warning(
            "enrichment delta: %d drop index(es) out of range [0, %d) — skipped",
            out_of_range_drop,
            n_facts,
        )

    # bindings{} — primary key is `bindings`; `new_entity_bindings` is a
    # legacy-shape synonym kept so older responses don't silently lose
    # the payload.  Each entry must be a non-empty string→string pair,
    # then routed through the shared table normalizer
    # (`placeholder_side="key"`, since a binding's canonical direction is
    # `{placeholder: real_text}` — the OPPOSITE of the CORE anonymizer
    # table) so an inverted binding (e.g. `{"Acme": "Org_9"}`, key/value
    # swapped) is corrected rather than passed straight into the
    # substitution map, and a binding where neither side is
    # placeholder-shaped is dropped rather than silently accepted.
    bindings: dict[str, str] = {}
    raw_bindings = parsed.get("bindings")
    if raw_bindings is None:
        raw_bindings = parsed.get("new_entity_bindings")
    if isinstance(raw_bindings, dict):
        str_bindings = {
            k: v
            for k, v in raw_bindings.items()
            if isinstance(k, str) and isinstance(v, str) and k and v
        }
        # `_normalize_anonymization_mapping` already logs a warning for
        # any binding it drops (both-or-neither shape match) — no second
        # log here.
        bindings, _bindings_stats = _normalize_anonymization_mapping(
            str_bindings, placeholder_side="key"
        )
    elif raw_bindings is not None:
        logger.warning(
            "enrichment delta: 'bindings' has non-dict shape %s — ignored",
            type(raw_bindings).__name__,
        )

    return add, modify, drop, bindings


def _reconstruct_updated_transcript(
    anon_transcript: str | None,
    bindings: dict[str, str],
) -> str | None:
    """Substitute SOTA-introduced bindings into the anonymized transcript.

    Replaces each binding's span with ``{{<placeholder>}}``, via
    :func:`~paramem.graph.placeholders._substitute_whole_words` — longest
    span first (so ``"Senior Software Engineer"`` wins over
    ``"Software Engineer"`` rather than being partially consumed by it)
    and edge-aware word-boundary matching, so ``"Bill"`` never matches
    inside ``"Billing"``.  All occurrences of each span are replaced —
    entities mentioned more than once in the transcript get one
    placeholder consistently.

    Returns the substituted transcript, the input unchanged when there
    are no bindings, or ``None`` when ``anon_transcript`` is ``None``.
    """
    if anon_transcript is None:
        return None
    if not bindings:
        return anon_transcript
    # Single-brace `{Prefix_N}` matches the convention SOTA used to echo
    # in the previous protocol's `updated_transcript` (saved snapshots
    # under `data/ha/debug/`) and the literal that `_apply_bindings`
    # already substitutes in fact subject / object.
    span_to_braced = {span: braced(placeholder) for placeholder, span in bindings.items()}
    return _substitute_whole_words(anon_transcript, span_to_braced)


def _apply_enrichment_delta(
    facts: list[dict],
    raw: str | None,
    anon_transcript: str | None,
) -> tuple[list[dict] | None, str | None, dict[str, str], dict]:
    """Apply the enrichment delta to input facts and reconstruct transcript.

    Returns ``(new_facts, updated_transcript, bindings, counts)``.  On
    parse failure ``new_facts`` is ``None``, which :func:`_filter_with_sota`
    reports to ``_sota_pipeline`` as a failed ``sota_enrich`` phase and
    which that call site turns into a raised
    :class:`ExtractionFailed` — an unparseable enrichment response fails
    the cycle rather than keeping the pre-enrichment facts.
    ``counts`` is a small diagnostic dict
    (``add_count`` / ``modify_count`` / ``drop_count`` / ``bindings_count``)
    that callers persist into ``graph.diagnostics``; on parse failure
    every count is zero.

    Application order:
      1. ``modify`` — shallow-merge ``fields`` into a copy of each
         indexed input fact.
      2. ``drop`` — remove dropped indices.
      3. ``add`` — append new facts.
      4. Reconstruct ``updated_transcript`` from ``anon_transcript`` +
         ``bindings`` (longest-span-first single pass).

    The transcript-on-the-wire is gone: SOTA emits only the bindings,
    and the substitution is deterministic.  Downstream diagnostics that
    used ``sota_updated_transcript`` continue to work because the
    reconstruction lives at the same call site.
    """
    parsed = _parse_enrichment_delta(raw, len(facts))
    if parsed is None:
        zero_counts = {
            "add_count": 0,
            "modify_count": 0,
            "drop_count": 0,
            "bindings_count": 0,
        }
        return None, None, {}, zero_counts
    add, modify, drop, bindings = parsed
    working = [dict(f) for f in facts]
    for idx, fields in modify:
        working[idx].update(fields)
    surviving = [f for i, f in enumerate(working) if i not in drop]
    surviving.extend(add)
    counts = {
        "add_count": len(add),
        "modify_count": len(modify),
        "drop_count": len(drop),
        "bindings_count": len(bindings),
    }
    return (
        surviving,
        _reconstruct_updated_transcript(anon_transcript, bindings),
        bindings,
        counts,
    )


def _plausibility_filter_with_sota(
    enriched_anon_facts: list[dict],
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    anon_transcript: str | None = None,
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
    prompts_dir: str | Path | None = None,
) -> tuple[list[dict] | None, str | None]:
    """SOTA plausibility filter — drops invalid relations only.

    No additions, no modifications. See sota_plausibility.txt for the
    drop criteria (self-loops, tautologies, role leaks, etc.).

    The judge emits a small ``{"drop": [<index>, ...]}`` object; this
    helper applies the drop-set to the input facts and returns the
    survivors.  Output is bounded and tiny by construction, so the
    truncation failure mode that hit the previous "echo every fact"
    protocol cannot recur on long inputs.

    Returns `(facts, raw_response)`. Raw response is preserved so callers
    can inspect the judge's verdict when questioning drop decisions.

    The prompt is external config — edit ``configs/prompts/sota_plausibility.txt``
    to tune; no code changes are needed.

    ``prompts_dir`` overrides the search directory (forwarded to
    :func:`_load_prompt`) so a calibration override actually reaches the
    judge; it defaults to the production template.
    """
    plaus_prompt = _load_prompt("sota_plausibility.txt", prompts_dir=prompts_dir, required=True)
    system_prompt = _load_prompt("sota_plausibility_system.txt", required=True)
    facts_json, transcript_text = _sota_facing_payload(enriched_anon_facts, anon_transcript)
    prompt = plaus_prompt.format(facts_json=facts_json, transcript=transcript_text)
    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
    )
    return _apply_drop_set(enriched_anon_facts, raw), raw


def local_plausibility_filter(
    facts: list[dict],
    transcript: str,
    model,
    tokenizer,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "sota_plausibility.txt",
) -> tuple[list[dict] | None, str]:
    """Local-model plausibility filter — drops invalid relations only.

    Same prompt as the SOTA plausibility filter, executed by a local model.
    Caller decides what data to pass: anonymized facts (placeholder strings)
    or de-anonymized facts (real names). The prompt is stage-agnostic.

    Returns ``(filtered_list, raw_output)``.  ``filtered_list`` is ``None``
    on parse failure (caller falls back).  The raw model output is the
    second element so calibration can capture it via phase_trace without
    re-running the call; an empty string indicates no raw response was
    obtained.

    The prompt is external config — edit ``configs/prompts/sota_plausibility.txt``
    to tune; no code changes are needed.

    ``seed`` is forwarded verbatim to :func:`generate_answer`.  At the
    default ``temperature=0.0`` (greedy decoding) it is a strict no-op.

    ``prompts_dir`` overrides the search directory and ``prompt_filename``
    the template name (both forwarded to :func:`_load_prompt`) so a
    calibration override actually reaches the model; both default to the
    production template.
    """
    _vram_snapshot(f"plaus_filter_entry n_facts={len(facts)}")
    plaus_prompt = _load_prompt(prompt_filename, prompts_dir=prompts_dir, required=True)
    system_prompt = _load_prompt("sota_plausibility_system.txt", required=True)
    prompt = plaus_prompt.format(
        facts_json=_render_indexed_facts(facts),
        transcript=transcript or "(not available)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )
    # Token count is the actual KV-cache driver, not character count.
    try:
        token_count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
    except Exception:  # noqa: BLE001
        token_count = -1
    logger.info(
        "plaus_filter prompt: chars=%d tokens=%d max_new_tokens=%d",
        len(formatted),
        token_count,
        max_tokens,
    )
    _vram_snapshot("plaus_filter_pre_generate")
    # vram_scope: the plausibility prompt drives a long generate (≥6 K
    # token prompt + up to 8 K new tokens of KV cache). Without this wrap,
    # the cached pool stays held when the next phase (procedural extraction
    # in consolidation.extract_session) starts its own prefill, and on an
    # 8 GiB device with STT/TTS resident the device exhausts. The scope's
    # finally clause runs torch.cuda.empty_cache() so the cached pool is
    # returned before control passes back to the caller.
    try:
        with vram_scope("plaus_filter"):
            raw = generate_answer(
                model,
                tokenizer,
                formatted,
                max_new_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "plaus_filter generate_answer raised %s: %s",
            type(exc).__name__,
            exc,
        )
        _vram_snapshot("plaus_filter_post_generate_error")
        raise
    _vram_snapshot("plaus_filter_post_generate")
    return _apply_drop_set(facts, raw), raw


# ---------------------------------------------------------------------------
# Predicate-normalization primitive
# ---------------------------------------------------------------------------


def normalize_predicates(
    relations: list[dict],
    *,
    model=None,
    tokenizer=None,
    sota: dict | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = 0.0,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "predicate_normalization.txt",
    system_filename: str = "predicate_normalization_system.txt",
) -> tuple[dict[tuple[str, str], list[list[str]]], dict]:
    """Backend-agnostic synonym-predicate clustering over same-(subject,object) groups.

    Groups ``relations`` by ``(canonical(subject), canonical(object))``.  Candidate
    groups are those with **≥2 distinct canonical predicates**.  Single-predicate
    groups pass through untouched with no model call.

    For each candidate group exactly ONE model call is made — with the predicate
    list for that group only (entities are never sent to the model).  The returned
    ``{"clusters": [...]}`` output is grounded against the group's actual input
    predicates; clusters with fewer than 2 grounded members are discarded.

    Two irreducible backend branches:

    * ``sota is None`` → LOCAL: ``model`` and ``tokenizer`` are required.
      Gradient checkpointing is disabled once before the candidate loop and
      re-enabled in ``finally``.  Each call is wrapped in
      ``vram_scope("dedup")``.
    * ``sota`` dict → CLOUD: ``_sota_call`` receives the RAW rendered prompt
      (no ``apply_chat_template``) and the ``normalization_system_prompt``
      prompt loaded here; a ``None`` return (network or parse failure) yields
      empty clusters for that group — facts are never deleted on failure.

    Both the normalization prompt (``prompt_filename``) and the system prompt
    (``system_filename``) load through :func:`_load_prompt` — the single
    production prompt-loading chokepoint — for BOTH branches, so a
    calibration override or provenance record reaches either backend
    identically.

    ``seed`` is forwarded verbatim to :func:`generate_answer`.  At the default
    ``temperature=0.0`` (greedy decoding) it is a strict no-op.

    Args:
        relations: Flat list of relation dicts (``subject``, ``predicate``,
            ``object`` keys at minimum).
        model: Local inference model.  Required when ``sota`` is ``None``.
        tokenizer: Local tokenizer.  Required when ``sota`` is ``None``.
        sota: Cloud backend configuration dict with keys ``api_key``,
            ``provider``, ``filter_model``, ``endpoint``.  When ``None`` the
            local branch is used.
        max_tokens: Maximum new tokens for each model call.
        temperature: Sampling temperature (0.0 = greedy).
        seed: RNG seed forwarded to :func:`generate_answer`.  No-op at
            ``temperature=0.0``.
        prompts_dir: Overrides the search directory for both prompts
            (forwarded to :func:`_load_prompt`).
        prompt_filename: Overrides the filter-prompt template name.
        system_filename: Overrides the system-prompt template name.

    Returns:
        clusters_by_so: ``dict[(canonical_s, canonical_o) -> list[list[str]]]``
            mapping each candidate group to its non-trivial synonym clusters
            (≥2 grounded members each).  Groups where the model returned no
            valid cluster are absent.  Caller owns any graph-apply and ledger
            operations.
        diagnostics: ``dict`` with keys ``groups_examined``,
            ``candidate_groups``, ``groups_with_clusters``, ``model_calls``,
            ``raw_outputs`` (list of per-group raw model strings), ``discards``.
    """
    from paramem.utils.identity import canonical

    clusters_by_so: dict[tuple[str, str], list[list[str]]] = {}

    if not relations:
        return clusters_by_so, {
            "groups_examined": 0,
            "candidate_groups": 0,
            "groups_with_clusters": 0,
            "model_calls": 0,
            "raw_outputs": [],
            "discards": [],
        }

    normalization_prompt = _load_prompt(prompt_filename, prompts_dir=prompts_dir, required=True)
    normalization_system_prompt = _load_prompt(
        system_filename, prompts_dir=prompts_dir, required=True
    )

    # --- Build canonical grouping -------------------------------------------
    groups: dict[tuple[str, str], list[dict]] = {}
    for rel in relations:
        s = str(rel.get("subject", ""))
        o = str(rel.get("object", ""))
        key = (canonical(s), canonical(o))
        groups.setdefault(key, []).append(rel)

    # Candidate groups: ≥2 distinct canonical predicates on the same (s, o).
    # Single-predicate groups pass through untouched (no model call).
    candidate_keys: list[tuple[str, str]] = [
        key
        for key, rels in groups.items()
        if len({canonical(str(r.get("predicate", ""))) for r in rels}) >= 2
    ]

    diagnostics: dict = {
        "groups_examined": len(groups),
        "candidate_groups": len(candidate_keys),
        "groups_with_clusters": 0,
        "model_calls": 0,
        "raw_outputs": [],
        "discards": [],
    }

    if not candidate_keys:
        return clusters_by_so, diagnostics

    local_mode = sota is None
    # Predicate-normalization is structured extraction: the local path must run on the
    # base weights (adapter disabled) with the KV cache live (checkpointing off,
    # restored to entry state on exit).  The SOTA path uses the cloud model and
    # leaves the local model untouched.
    cm = base_model_inference(model) if local_mode else contextlib.nullcontext()
    with cm:
        for key in candidate_keys:
            rels_for_group = groups[key]
            # Build canonical → first-seen surface predicate map.
            pred_surface: dict[str, str] = {}
            for rel in rels_for_group:
                can_pred = canonical(str(rel.get("predicate", "")))
                if can_pred not in pred_surface:
                    pred_surface[can_pred] = str(rel.get("predicate", ""))
            preds_to_send = list(pred_surface.values())

            rendered = normalization_prompt.format(predicates_json=json.dumps(preds_to_send))

            if local_mode:
                messages = [
                    {"role": "system", "content": normalization_system_prompt},
                    {"role": "user", "content": rendered},
                ]
                formatted = tokenizer.apply_chat_template(
                    adapt_messages(messages, tokenizer),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                with vram_scope("dedup"):
                    raw = generate_answer(
                        model,
                        tokenizer,
                        formatted,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                    )
            else:
                raw = _sota_call(
                    rendered,
                    api_key=sota["api_key"],
                    provider=sota["provider"],
                    filter_model=sota["filter_model"],
                    endpoint=sota["endpoint"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=normalization_system_prompt,
                )
                if raw is None:
                    diagnostics["raw_outputs"].append("")
                    continue

            diagnostics["model_calls"] += 1
            diagnostics["raw_outputs"].append(raw)
            logger.debug("dedup group %r raw: %s", key, raw[:300] if raw else "")

            # Parse {"clusters": [["predA", "predB"], ...]} schema.
            raw_clusters: list = []
            try:
                json_str = _extract_json_block(raw)
                data = json.loads(json_str)
                if isinstance(data, dict):
                    raw_clusters = data.get("clusters", [])
                if not isinstance(raw_clusters, list):
                    raw_clusters = []
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("dedup parse failed for group %r: %s", key, exc)

            # Ground cluster members against input predicates for this group.
            input_can_preds = set(pred_surface.keys())
            grounded_clusters: list[list[str]] = []
            for cluster in raw_clusters:
                if not isinstance(cluster, list):
                    continue
                grounded = [
                    can_p
                    for p in cluster
                    for can_p in (canonical(str(p)),)
                    if can_p in input_can_preds
                ]
                # Always record hallucinated predicates (those not in input) in
                # discards — regardless of whether the cluster passes the ≥2
                # grounded check.
                for p in cluster:
                    can_p = canonical(str(p))
                    if can_p not in input_can_preds and p:
                        diagnostics["discards"].append(
                            {
                                "reason": "hallucinated_predicate",
                                "predicate": str(p),
                                "group": list(key),
                            }
                        )
                if len(grounded) >= 2:
                    grounded_clusters.append(grounded)

            if grounded_clusters:
                clusters_by_so[key] = grounded_clusters
                diagnostics["groups_with_clusters"] += 1

    return clusters_by_so, diagnostics
