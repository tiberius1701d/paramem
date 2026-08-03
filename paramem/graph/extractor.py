"""LLM-based knowledge graph extraction — generate once, parse once."""

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from paramem.cloud.admission import OPENAI_COMPAT_ENDPOINTS, OPENAI_COMPAT_PROVIDERS
from paramem.cloud.anonymize import AnonymizedContract
from paramem.cloud.deanonymize import (
    CloudScope,
    DeanonResult,
    _extract_json_block,
    deanonymize_facts,
    deanonymize_text,
)
from paramem.cloud.placeholders import (
    _FACT_FIELDS,
    _fact_orphans,
    _fact_tokens,
    _normalize_anonymization_mapping,
    _substitute_whole_words,
    braced,
    insert_placeholders,
)
from paramem.config.taxonomy import (
    fallback_entity_type,
    fallback_relation_type,
    relation_types,
)
from paramem.evaluation.recall import generate_answer
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.graph.prompts import _load_prompt
from paramem.graph.relation_build import build_relations
from paramem.graph.schema import SessionGraph
from paramem.models.loader import adapt_messages, base_model_inference
from paramem.utils.identity import as_speaker_id, canonical, is_speaker_id
from paramem.utils.tokens import estimate_tokens
from paramem.utils.vram_guard import vram_scope

logger = logging.getLogger(__name__)


class ExtractionFailed(RuntimeError):
    """Raised when a load-bearing extraction phase fails and the cycle
    must be aborted for this session.

    Currently raised from the ``cloud_enrich`` phase when the cloud
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
    ``"cloud_enrich"``).  ``reason`` is a short operator-facing string.
    """

    def __init__(self, phase: str, reason: str) -> None:
        super().__init__(f"{phase}: {reason}")
        self.phase = phase
        self.reason = reason


# Single output-token budget for every LLM call in the extraction pipeline:
# local extraction, anonymization, cloud enrichment, plausibility (local +
# cloud), graph-level enrichment. Threaded through
# ``paramem.graph.flows.extract_graph`` → the ``anonymize``/``enrich``
# stages (``paramem.graph.stage_anonymize`` / ``paramem.graph.stage_enrich``) → all
# sub-functions so a single ``ConsolidationLoop.extraction_max_tokens``
# (server.yaml ``consolidation.extraction_max_tokens``) governs the whole
# chain.
#
# 8192 is sized for Mistral 7B against document chunks up to
# ``paramem.graph.document_chunker._DOC_MAX_TOKENS``, the local chunker's
# max — currently ~828 words (_DOC_MAX_TOKENS is DERIVED from the
# anonymize-call token envelope rather than an independent ~1500-word
# heuristic, and that derivation itself consumes the empirical figure
# below). Empirical worst-case observed output for a dense resume chunk was
# ~2200 tokens against the PRIOR ~1500-word max; 8192 still gives ample
# headroom against the smaller current max (a smaller chunk produces
# proportionally less extraction output, not more). If the chunker contract
# changes again, revisit jointly with that change — this comment AND
# ``_DOC_MAX_TOKENS``'s own derivation comment both consume the ~2200-token
# figure and must be updated together.
#
# Plausibility output couples to chunk density. The filter's contract
# (configs/prompts/cloud_plausibility.txt) is "Return ONLY a JSON array of
# surviving facts, schema unchanged" — so its output volume scales with
# the surviving-fact count, which scales with chunk density. Lowering the
# cap independently for plausibility was attempted and reverted: a 2048
# cap truncated the JSON array on dense chunks, the parse failed, and the
# caller fell back to passing the unfiltered set forward. KV-cache
# pressure must be mitigated upstream (STT/TTS eviction, gc.collect
# before empty_cache, per-phase vram_scope wraps), not by truncating
# correctness-bearing output.
_DEFAULT_FILTER_MAX_TOKENS = 8192
# Deterministic by default; threaded to every provider call so Anthropic
# and OpenAI-compatible filters match exactly.
_DEFAULT_FILTER_TEMPERATURE = 0.0
# Cloud enrichment returns the delta envelope ``{"add", "modify", "drop",
# "bindings"}`` on virtually every call; a bare-list / malformed shape is a
# rare per-sample deviation (2 in the whole recorded history vs. dozens of
# clean deltas).  Re-issue the call on a parse miss rather than accept a
# guessed-at shape — a fresh sample almost always parses.  Exhausting the
# budget hands ``None`` back to the caller, which fails OPEN (keeps the
# pre-enrichment facts), never fatal.
_ENRICHMENT_MAX_ATTEMPTS = 3
# Per-call timeout for cloud enrichment / plausibility / anonymization.
_DEFAULT_FILTER_TIMEOUT_SECONDS = 90.0


# ---------------------------------------------------------------------------
# WSL2 GPU wake helper — covers the post-cloud-call → next-GPU-op gap.
#
# Background: WSL2 + RTX 5070 + Modern Standby lets the GPU enter a low-power
# state after ~60s of idle. A cloud round-trip is a typical trigger
# (anonymization completes → cloud cloud call takes 30–90s → next local-LLM
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

    Used to localise VRAM-pressure-induced crashes in the cloud pipeline.
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
               extraction prompts are per-model; cloud prompts call
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
            ``configs/prompts/``.  cloud prompts are not affected.
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
    extra_slots: dict[str, str] | None = None,
    postprocess: Callable[[SessionGraph], SessionGraph] | None = None,
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

    Args:
        extra_slots: Additional ``.format()`` kwargs for the user prompt
            template, beyond the always-supplied ``transcript`` and
            ``speaker_context``. ``None`` (default) supplies none — every
            existing caller is unaffected. ``second_order_extract`` passes
            ``{"named_people": ...}`` to thread its gate-derived closed
            target set into ``extraction_second_order.txt``'s
            ``{named_people}`` slot.
        postprocess: Optional caller-supplied enforcement hook, applied to
            the freshly-parsed graph BEFORE :func:`_summarise_graph` runs
            for the phase trace — so the trace's ``parsed`` snapshot
            reflects the graph the caller actually keeps, not the raw
            parse. ``None`` (default) applies no enforcement; every
            existing caller is unaffected.
            ``second_order_extract`` (:func:`~paramem.graph.flows.
            _stage_second_order_extract`) is the one caller that supplies
            one — its off-target-subject remap/drop enforcement — so the
            ``second_order_extract`` phase's trace counts match what
            actually reaches ``graph.relations``/``graph.entities``. Never
            called on the parse-failure path (the caller has nothing to
            enforce over an empty graph).
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
            extra_slots=extra_slots,
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
        if postprocess is not None:
            graph = postprocess(graph)
        t.set_parsed(_summarise_graph(graph))
    return graph


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
    extra_slots: dict[str, str] | None = None,
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
    :func:`load_extraction_prompts`.  The ``cloud_*`` prompts are
    model-independent by design and are not affected.

    ``seed`` is forwarded verbatim to :func:`generate_answer`.  At the
    default ``temperature=0.0`` (greedy decoding) it is a strict no-op.

    ``vram_label`` names the :func:`~paramem.utils.vram_guard.vram_scope`
    wrap around the generate call. Defaults to ``"extract_main"``
    (``local_extract``/``second_order_extract``); callers with a distinct
    VRAM-telemetry identity (e.g. ``extract_procedural_graph`` using
    ``"procedural"``) override it.

    ``extra_slots`` supplies additional **user**-template ``.format()``
    kwargs alongside ``transcript``/``speaker_context`` — e.g.
    ``second_order_extract`` supplies ``{"named_people": ...}`` for
    ``extraction_second_order.txt``'s ``{named_people}`` slot. ``None``
    (default) adds nothing; a caller whose template references a slot not
    supplied here or by ``extra_slots`` raises ``KeyError`` at
    ``.format()`` time. A key in ``extra_slots`` that collides with
    ``transcript``/``speaker_context`` raises ``TypeError`` (``dict()``'s
    own "got multiple values for keyword argument" — the kwargs dict is
    built as ``dict(transcript=..., speaker_context=..., **extra_slots)``)
    rather than silently overwriting either always-supplied slot.
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
        **(extra_slots or {}),
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
    rewrapped here so downstream normalization can proceed. A model that
    wraps the whole envelope dict in a one-element list (``[{"entities":
    [...], "relations": [...]}]``, observed live) never reaches this
    function as a list at all — :func:`~paramem.cloud.deanonymize.
    _extract_json_block` unwraps that shape to the inner envelope dict's
    JSON at the shared parsing primitive, the one boundary every
    structured-output parser in this module shares.

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
        # A list here is always a bare fact list — _extract_json_block
        # already unwrapped the "whole envelope wrapped in a one-element
        # list" shape to the inner dict, so `data` is a dict in that case.
        # Wrap as a relations payload; _normalize_extraction walks
        # ``relations`` and infers the entity set from subject/object.
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
         ``len(canonical(speaker_name).split()) >= 2``.  ``canonical``
         emits a space as the blank, so the name parts are the
         space-separated tokens of the canonical form; ``-`` is not a blank,
         so a hyphenated single given name ("Anna-Maria") counts as ONE part. A single-token
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
    # family), never a bare given name.  ``canonical`` emits a space as the
    # blank, so the name parts are the space-separated tokens of the
    # canonical form.  ``-`` is not a blank, so a hyphenated single given name
    # ("Anna-Maria") counts as ONE part and does not open the rewrite.
    do_rewrite = (
        source_type == "document"
        and speaker_name
        and not is_speaker_id(speaker_name)
        and len(canonical(speaker_name).split()) >= 2
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
# paramem.cloud.deanonymize (imported above) — every cloud-response
# parser in this module still routes through the one shared parser.


# Fallbacks resolved per-call via paramem.config.taxonomy.


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
            # Ingest safety-net: resolve any speaker-id token to its canonical
            # form at the single normalization boundary.  Extraction prompts
            # instruct the model to emit lowercase speaker{N} directly; this
            # resolves any residual cased form ("Speaker0", "SPEAKER0") a model
            # emits despite the instruction.  ONLY speaker-id tokens are
            # resolved — display names are untouched.
            name_sid = as_speaker_id(norm["name"])
            if name_sid is not None:
                norm["name"] = name_sid
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
            # Ingest safety-net: resolve any speaker-id token to its canonical
            # form at this boundary.
            subj_sid = as_speaker_id(subject)
            if subj_sid is not None:
                subject = subj_sid
            obj_sid = as_speaker_id(obj)
            if obj_sid is not None:
                obj = obj_sid

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


# Two-stage cloud pipeline: enrichment first, then plausibility filtering.
# Each stage has a single responsibility and a separate prompt — combining
# them in one call (the previous "enrichment_provider" prompt) led to the LLM
# expanding scope at the same time as filtering, producing inflated counts
# and self-referential schema artifacts.

# The cloud plausibility judge — a judge that only ever sees anonymized
# data — dispatches on the SAME provider registry as enrichment
# (PROVIDER_KEY_ENV in paramem.cloud.admission).  ``plausibility_judge``
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
    speaker_id: str,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    seed: int | None = None,
) -> SessionGraph:
    """Fallback pipeline path: run local plausibility on raw (unanonymized) facts.

    Used when anonymization fails entirely (mapping parse failure — no safe
    Cloud path), or when the full pipeline drops all relations.

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
        filtered, _raw = judge_plausibility(
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
    """Persist a :func:`~paramem.cloud.deanonymize.deanonymize_facts`
    result's collision findings onto ``graph.diagnostics``.

    The gate primitive
    (:func:`~paramem.cloud.placeholders._binding_collisions`) used to
    write these keys itself, from inside ``deanonymize_facts``, onto a
    ``SessionGraph`` it took purely as a diagnostics sink; a caller two
    levels up then read the mutation back off the graph. The finding is a
    return value now, and this is the ONE place in the extractor that
    turns it into a diagnostic, shared by both remaining
    ``deanonymize_facts`` call sites (the ``deanon`` substitution and
    :func:`request_graph_enrichment`; the former ``cloud_enrich`` gate
    call site was retired along with the whole-delta rejection it existed
    for — see :func:`_apply_enrichment_delta`).

    **``cloud_pending_orphans`` is retired (2026-07-22 cloud-admission
    redesign).** ``DeanonResult`` no longer carries a ``verdict`` field —
    nothing gates on a whole-delta orphan list any more, so there is
    nothing left to write under that key. Only ``collisions`` survives:
    always informational (a binding for a token cloud was shown is inert
    under CORE-LAST precedence), never a rejection signal.

    Writes are guarded exactly as the primitive's were: an EMPTY list
    writes no key at all, so ``"cloud_binding_collisions" not in
    graph.diagnostics`` keeps its established meaning ("the scan found
    nothing"), distinct from a present-but-empty value.

    Args:
        graph: The graph the delta is being applied to — the session graph
            for session-tier extraction, the caller's throwaway per-chunk
            graph for graph-tier enrichment.
        result: The ``DeanonResult`` just returned for that delta.
    """
    if result.collisions:
        graph.diagnostics["cloud_binding_collisions"] = result.collisions


# The provider tables (PROVIDER_KEY_ENV, OPENAI_COMPAT_ENDPOINTS,
# OPENAI_COMPAT_PROVIDERS) and the key resolver now live in
# paramem.cloud.admission (imported above), alongside
# evaluate_cloud_egress — the one place the "may we reach a cloud LLM,
# and with what credentials?" decision is computed.  They are imported
# here for dispatch (_cloud_call / _filter_openai_compat), not owned here.


# The three cloud system prompts below (``cloud_enrichment_system.txt``,
# ``cloud_plausibility_system.txt``, ``cloud_graph_enrichment_system.txt``)
# are loaded at CALL TIME inside each consuming function (``request_enrichment``,
# ``request_graph_enrichment``, ``request_plausibility``,
# ``judge_plausibility``) — never as module-level constants.  A
# module-level ``_load_prompt(...)`` call runs at import time, before any
# :func:`~paramem.graph.phase_trace.extraction_trace`/``phase_trace`` scope
# can exist and before :func:`~paramem.graph.prompts.prompt_overrides` can
# be active, so it is permanently unreachable by both prompt provenance and
# calibration overrides.  ``_filter_anthropic`` / ``_filter_openai_compat`` /
# ``_cloud_call`` still default their own ``system_prompt`` parameter to the
# enrichment prompt for callers that omit it (resolved lazily, in the body,
# never as a default-expression — a default expression is itself evaluated
# at function-definition time, i.e. import time, so it has the exact same
# unreachability problem as a module-level constant).


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
    calibration tool to probe cloud non-determinism.  Anthropic's API does
    not accept a ``seed`` parameter so seed-based reproducibility cannot
    be requested at this layer; the calibration tool reports
    ``params_effective.seed=null`` for cloud stages so the operator knows
    it was dropped.  Both default to ``None`` — production paths
    preserve current temperature-only sampling behaviour.

    ``system_prompt`` defaults to ``None``, resolved here (never as a
    default-expression, which would evaluate at import time) to the
    ``cloud_enrichment.txt`` system prompt — production callers
    (:func:`_cloud_call`, forwarded from :func:`request_enrichment`) always
    pass their own resolved prompt explicitly; the fallback here only
    serves a direct caller (e.g. a test) that omits it.
    """
    if system_prompt is None:
        system_prompt = _load_prompt("cloud_enrichment_system.txt", required=True)
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping cloud filter")
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
    ``cloud_enrichment.txt`` system prompt — see :func:`_filter_anthropic`
    for why.
    """
    if system_prompt is None:
        system_prompt = _load_prompt("cloud_enrichment_system.txt", required=True)
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — skipping cloud filter")
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


def _cloud_call(
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
    """Generic cloud dispatch (anthropic native or any OpenAI-compatible host).

    ``system_prompt`` defaults to ``None`` and is forwarded as-is to
    ``_filter_anthropic``/``_filter_openai_compat`` — each resolves its own
    ``None`` there (never here as a default-expression, which would
    evaluate at import time); production callers
    (:func:`request_enrichment`, :func:`request_graph_enrichment`,
    :func:`request_plausibility`) always pass their own resolved
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
    logger.warning("Unsupported cloud provider '%s'", provider)
    return None


def _parse_facts_response(raw: str | None, strict_array: bool = False) -> list[dict] | None:
    """Parse a cloud response into a list of fact dicts. Returns None on failure.

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
    logger.debug("cloud response raw: %s", raw[:500])
    try:
        json_str = _extract_json_block(raw)
        validated = json.loads(json_str)
        if isinstance(validated, list):
            return validated
        if not strict_array and isinstance(validated, dict):
            for key in ("relations", "filtered", "facts", "results"):
                if key in validated and isinstance(validated[key], list):
                    return validated[key]
        logger.warning("cloud response unexpected format: %s", type(validated).__name__)
        return None
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.debug("cloud response strict parse failed: %s — attempting salvage", e)
        salvaged = _salvage_fact_objects(raw)
        if salvaged:
            logger.warning(
                "cloud response strict parse failed (%s); salvaged %d fact dict(s) "
                "via stream-parse — likely a truncated array",
                e,
                len(salvaged),
            )
            return salvaged
        logger.warning("cloud response parse failed: %s", e)
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
    the cloud-style ``new_entity_bindings`` sub-dict into a fact list.
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


def request_enrichment(
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
    prompt_filename: str = "cloud_enrichment.txt",
    *,
    speaker_id: str,
) -> tuple["EnrichmentDelta | None", str | None, dict]:
    """Cloud enrichment call — coreference + compound splitting + safe
    reification.

    Returns ``(delta, raw_response, info)``. This function only calls the
    cloud and PARSES the response into an :class:`EnrichmentDelta` — it no
    longer applies the delta, merges facts, or reconstructs the updated
    transcript (2026-07-22 cloud-admission redesign). The caller applies
    the delta itself via :func:`_apply_enrichment_delta`, once it has built
    the :class:`~paramem.cloud.deanonymize.CloudScope` the delta's
    ``bindings`` need to be checked against.

    The cloud emits a delta envelope ``{"add": [...], "modify": [...],
    "drop": [...], "bindings": {...}}`` describing what to change against
    the indexed input facts. KEEP is the default; unnamed input facts pass
    through unchanged.

    ``delta.bindings`` maps each new braced placeholder cloud introduced
    (key without braces, e.g. ``"Event_1"``) to the exact transcript span
    it stands for. Cloud already knows the binding the moment it mints
    each placeholder, so emitting it explicitly removes the
    transcript-diff reconstruction step the previous "echo every fact"
    protocol relied on.

    ``info`` is a dict with diagnostic flags the caller persists into
    ``graph.diagnostics``:

    * ``parse_path``: ``"delta"`` (envelope parsed), ``"failed"`` (the
      provider answered but never in the delta-envelope shape on any
      attempt — a rare per-input hiccup), or ``"no_response"`` (the
      provider was unreachable on every attempt — an outage; ``_cloud_call``
      collapses every API/network/SDK error to ``None``).  The call retries
      up to ``_ENRICHMENT_MAX_ATTEMPTS`` times — the model returns the
      envelope on nearly every call, so a fresh sample almost always
      parses.  ``delta=None`` only after all attempts miss, and the caller
      (the ``enrich`` stage,
      :func:`~paramem.graph.stage_enrich._stage_enrich`) branches on
      ``parse_path``: ``"failed"`` fails OPEN (keeps the pre-enrichment
      facts, records a ``cloud_enrichment_degraded`` diagnostic — one
      session un-enriched, the run continues), while ``"no_response"``
      raises :class:`ExtractionFailed` so the batch aborts and its sessions
      stay pending for a clean retry once the provider recovers.
    * ``attempts``: how many cloud calls were issued (``1`` on the common
      first-try success, up to ``_ENRICHMENT_MAX_ATTEMPTS``).
    * ``response_chars``: length of the last raw response in characters.

    The action counts (``add_count`` / ``modify_count`` / ``drop_count`` /
    ``bindings_count``) that used to live in this dict moved to
    :func:`_apply_enrichment_delta`'s ``report`` — they depend on
    resolvability, which this function no longer decides.

    The prompt this function loads is external config — edit
    ``configs/prompts/cloud_enrichment.txt`` to tune; no code changes are
    needed.

    ``prompts_dir`` overrides the search directory (forwarded to
    :func:`_load_prompt`) so a calibration override actually reaches the
    model; it defaults to the production template. ``prompt_filename``
    overrides the file name within that directory (the production caller,
    the ``enrich`` stage, keeps the default); consistent with the
    ``prompt_filename`` parameter on
    :func:`~paramem.cloud.anonymize.anonymize_transcript` and
    :func:`judge_plausibility`.

    ``speaker_id`` fills the prompt's ``{speaker_id}`` slot — THIS
    session's own speaker anchor (e.g. ``"speaker0"``, ``"speaker1"``),
    so the "Speaker identity" binding in ``cloud_enrichment.txt`` names
    the correct household member as the transcript's ``[user]`` instead
    of hardcoding ``speaker0`` for every session (the multi-speaker
    session binding fix). Keyword-required, no default — a silently
    degraded prompt (a blank speaker anchor) is worse than a loud
    ``TypeError`` at the one production call site. The production caller
    (the ``enrich`` stage, :func:`~paramem.graph.stage_enrich._stage_enrich`)
    always threads ``ctx.speaker_id``, itself a required, non-empty field
    (:class:`~paramem.graph.flow.StageContext`).
    """
    enrichment_prompt = _load_prompt(prompt_filename, prompts_dir=prompts_dir, required=True)
    system_prompt = _load_prompt("cloud_enrichment_system.txt", required=True)
    facts_json, transcript_text = _cloud_facing_payload(anon_facts, anon_transcript)
    prompt = enrichment_prompt.format(
        facts_json=facts_json, transcript=transcript_text, speaker_id=speaker_id
    )
    last_raw: str | None = None
    responded = False
    for attempt in range(1, _ENRICHMENT_MAX_ATTEMPTS + 1):
        raw = _cloud_call(
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
            # No usable response — an API/network/SDK error, which
            # ``_filter_anthropic`` / ``_filter_openai_compat`` collapse to
            # ``None`` (see their ``except Exception`` boundary handlers).
            logger.warning(
                "cloud enrichment: no response on attempt %d/%d — retrying",
                attempt,
                _ENRICHMENT_MAX_ATTEMPTS,
            )
            continue
        responded = True
        last_raw = raw
        delta = _parse_enrichment_delta(raw, len(anon_facts))
        if delta is not None:
            return (
                delta,
                raw,
                {
                    "response_chars": len(raw),
                    "parse_path": "delta",
                    "attempts": attempt,
                },
            )
        # Parseable JSON but not the delta envelope (the rare bare-list /
        # malformed shape).  Log the actual payload so the deviation is
        # diagnosable — the type alone hides which shape the model returned —
        # then retry: the envelope is what it emits on nearly every call.
        logger.warning(
            "cloud enrichment delta parse failed on attempt %d/%d "
            "(response_chars=%d) — retrying; raw[:500]=%r",
            attempt,
            _ENRICHMENT_MAX_ATTEMPTS,
            len(raw),
            raw[:500],
        )
    # Every attempt missed.  ``responded`` is the OUTAGE-vs-HICCUP
    # discriminator: the provider answered at least once (``"failed"`` — a
    # per-input shape hiccup, caller fails OPEN) or it was unreachable on
    # every attempt (``"no_response"`` — an outage, caller leaves the batch
    # pending for a clean retry next cycle).
    if responded:
        return (
            None,
            last_raw,
            {
                "response_chars": len(last_raw or ""),
                "parse_path": "failed",
                "attempts": _ENRICHMENT_MAX_ATTEMPTS,
            },
        )
    return (
        None,
        None,
        {
            "response_chars": 0,
            "parse_path": "no_response",
            "attempts": _ENRICHMENT_MAX_ATTEMPTS,
        },
    )


# ---------------------------------------------------------------------------
# Graph-level cloud enrichment (Task #10)
# ---------------------------------------------------------------------------


def request_graph_enrichment(
    payload: AnonymizedContract,
    graph: SessionGraph,
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> tuple[list[dict], list[list[str]], str | None, int] | None:
    """Cloud graph-level enrichment pass over a pre-merged cumulative graph.

    Sends a subgraph serialized as triples to a cloud provider and requests
    two outputs:
    - New cross-session second-order relations not already in the graph.
    - ``same_as`` pairs identifying duplicate nodes under different surface forms.

    Runs the SAME anonymize -> cloud -> de-anonymize contract as the
    session-tier ``anonymize``/``enrich`` stages
    (:mod:`paramem.graph.stage_anonymize` / :mod:`paramem.graph.stage_enrich`), via
    the SAME chain in :mod:`paramem.cloud.anonymize` /
    :mod:`paramem.cloud.deanonymize` — this is the second call site of
    that contract. ``payload`` is (A)'s output for this chunk — the
    caller (:func:`~paramem.training.graph_enrich.enrich_graph`)
    already ran :func:`~paramem.cloud.anonymize.anonymize` (with
    ``identity_domain`` reconciliation and the domain-scoped fail-closed
    guard) before calling this function; this function applies no scope
    gate of its own — it only substitutes and de-anonymizes.

    ``payload.reverse`` is produced exclusively by
    :func:`~paramem.cloud.placeholders._build_anonymization_mapping`
    inside (A) — the speaker-value guard applies here by construction; no
    code path in this function inverts an unfiltered forward map.

    The chunk's anonymized ``subject``/``object`` fields come from
    :func:`~paramem.cloud.placeholders.insert_placeholders` applied
    DIRECTLY to ``payload.facts`` — the (real-name, un-substituted) triple
    subset :func:`~paramem.cloud.anonymize.anonymize` already cleared for
    egress (the caller's ``triples`` minus any fail-closed slice's triples;
    see that function's docstring) — never a ``Relation`` round trip through
    ``graph.relations`` — the caller's throwaway per-chunk
    ``SessionGraph`` no longer carries relations at all; see ``graph``'s
    own docstring entry below): every other key on each triple dict
    (``predicate``, ``relation_type``, ``speaker_id``) is copied VERBATIM
    — a speaker id can never be a mapping key (the anonymization prompt
    instructs the model to leave ``speaker{N}`` ids verbatim), so no
    substitution was ever needed there.

    After the cloud call, the response's ``relations`` are de-anonymized via
    :func:`~paramem.cloud.deanonymize.deanonymize_facts` (the single exit
    gate: predicate invariant, substitute, residual sweep, fail-closed) —
    returning real node names and bare ``speaker{N}`` ids, exactly what
    :func:`~paramem.training.graph_enrich.enrich_graph`'s existing
    consumption logic (including the speaker-pair guard that rejects a same_as pair
    where both surfaces are speaker ids) already expects. There is no
    whole-chunk rejection: every ``relations`` entry is effectively an
    ``add`` (this tier has no local baseline to preserve), so an
    individually-unresolvable relation is simply dropped by the residual
    sweep — the caller reads how many were dropped from the fourth tuple
    element and accumulates it into its own ``dropped_relations`` stat
    (see :func:`~paramem.training.graph_enrich.enrich_graph`'s
    docstring).
    Any ``bindings`` the response carries (cloud-minted placeholders —
    normally empty, since the prompt forbids inventing new nodes) are
    normalized inside :meth:`~paramem.cloud.deanonymize.CloudScope.
    response` (placeholder on the key side) before being folded into
    that substitution, mirroring the session tier. ``same_as`` pairs are
    restored per-member via
    :func:`~paramem.cloud.deanonymize.deanonymize_text` — a pair
    with either member unresolved (``None``) is dropped.

    ``deanonymize_text`` deliberately does NOT run the
    undeclared-orphan SHAPE backstop (:data:`~paramem.cloud.placeholders.
    PLACEHOLDER_TOKEN_RE`) that :func:`~paramem.cloud.placeholders.
    _apply_bindings` runs for fact fields — see that function's docstring
    for why (free conversational prose is a wide false-positive surface
    for a bare shape match, e.g. ``GPT_4``, ``COVID_19``). That rationale
    does NOT apply to a ``same_as`` member — it is a structured node-name
    field, not prose — but the omission is still safe HERE specifically,
    for a different reason: an undeclared placeholder-shaped token that
    survives ``deanonymize_text`` unresolved (e.g. a stray
    ``"Person_99"`` cloud invented) is never itself a real node in this
    process's graph, so
    :func:`~paramem.training.graph_enrich.enrich_graph`'s own
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

    Loads ``cloud_graph_enrichment.txt`` (required). The prompt uses a
    ``{triples_json}`` placeholder.

    Args:
        payload: :class:`~paramem.cloud.anonymize.AnonymizedContract` —
            the caller's already-completed (A) result for this chunk.
            ``payload.facts`` is the source of the triples this function
            sends to cloud — no separate ``triples`` argument; a
            fail-closed slice's triples never reach this function because
            they never survived into ``payload.facts``.
        graph: The caller's throwaway per-chunk ``SessionGraph`` (carries
            no relations of its own — this function never reads
            ``graph.relations``) — the diagnostics sink this function
            writes the collision findings to (via
            :func:`_record_binding_diagnostics`).
        api_key: Provider API key.
        provider: Cloud provider name (e.g. ``"anthropic"``).
        filter_model: Model identifier for the provider.
        endpoint: Custom endpoint for OpenAI-compatible providers.
        max_tokens: Maximum tokens in the cloud response.
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        ``(new_relations, same_as_pairs, raw_response, dropped_relations)``
        on success, or ``None`` when the cloud call fails or the response
        cannot be parsed.  ``new_relations`` is a list of relation dicts
        with real node names; ``same_as_pairs`` is a list of
        ``[canonical, variant]`` pairs with real node names / bare speaker
        ids.  ``dropped_relations`` is the count of relations the
        fail-closed residual sweep individually dropped post-substitution
        (predicate-invariant drops plus residual-placeholder drops) — ``0``
        when every relation survived.  A legitimately empty response
        (``([], [], raw_response, 0)``) is indistinguishable from "nothing
        was dropped" by design — there is no whole-chunk rejection left to
        discriminate from an empty delta.

    The prompt this function loads is external config — edit
    ``configs/prompts/cloud_graph_enrichment.txt`` to tune; no code changes
    are needed.
    """
    anon_triples = insert_placeholders(payload.facts, payload.forward)

    enrichment_prompt = _load_prompt("cloud_graph_enrichment.txt", required=True)
    system_prompt = _load_prompt("cloud_graph_enrichment_system.txt", required=True)
    # No try/except: a KeyError here means the prompt template has an
    # un-doubled literal brace (a template bug, not a runtime condition).
    # Swallowing it turned a missed brace-doubling into a permanent,
    # silent outage of graph enrichment — it must kill the fold loudly.
    triples_json = json.dumps(anon_triples, indent=2)
    prompt = enrichment_prompt.format(triples_json=triples_json)

    raw = _cloud_call(
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
    # triples_json string sent to cloud (never a shape scrape).
    scope = CloudScope.response(payload, cloud_bindings=raw_bindings, sent=(triples_json,))

    # De-anonymize relations — the SAME exit gate the session tier uses.
    # Every ``new_relations`` entry here is effectively an ``add`` (this
    # tier has no local baseline to preserve — ``graph`` carries no
    # relations of its own), so there is no whole-chunk gate any more: an
    # individually-unresolvable relation is simply dropped by
    # ``_apply_bindings``'s fail-closed residual sweep (predicate
    # invariant + residual placeholder check), never a reason to discard
    # every OTHER relation and ``same_as`` pair in the same response.
    deanon = deanonymize_facts(scope, new_relations)
    _record_binding_diagnostics(graph, deanon)
    dropped_relation_count = len(deanon.predicate_dropped) + len(deanon.residual_dropped)
    if dropped_relation_count:
        logger.warning(
            "graph_enrichment: dropped %d relation(s) post-substitution "
            "(%d predicate-invariant, %d residual placeholder sweep).",
            dropped_relation_count,
            len(deanon.predicate_dropped),
            len(deanon.residual_dropped),
        )

    # same_as pairs: per-member free-text deanon — a pair with either
    # member unresolved (declared-but-unobserved, or otherwise fail-closed)
    # is dropped rather than forwarded with a residual placeholder.
    deanon_same_as: list[list[str]] = []
    for canon, variant in same_as_pairs:
        d_canon = deanonymize_text(scope, canon)
        d_variant = deanonymize_text(scope, variant)
        if d_canon is None or d_variant is None:
            logger.warning(
                "graph_enrichment: dropping same_as pair with unresolved token: %r",
                [canon, variant],
            )
            continue
        deanon_same_as.append([d_canon, d_variant])

    return deanon.facts, deanon_same_as, raw, dropped_relation_count


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


def _cloud_facing_payload(facts: list[dict], anon_transcript: str | None) -> tuple[str, str]:
    """The exact ``(facts_json, transcript_text)`` pair rendered into
    every cloud-facing prompt.

    ONE render so :func:`request_enrichment`'s enrichment prompt,
    :func:`request_plausibility`'s prompt, and the ``enrich``
    stage's (:func:`~paramem.graph.stage_enrich._stage_enrich`) ``observed``
    legality-domain scan (the set of placeholder tokens cloud was actually
    shown) cannot drift from one another — previously that invariant was
    enforced only by a code comment next to a hand-mirrored copy of the
    render.
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
    (matches the prior contract: ``filtered_list is None`` → the caller
    (e.g. the ``enrich`` stage) keeps all input facts unchanged and logs a
    warning).  Empty drop set → input list returned unchanged.
    """
    drop = _parse_drop_set(raw, len(facts))
    if drop is None:
        return None
    if not drop:
        return list(facts)
    return [f for i, f in enumerate(facts) if i not in drop]


@dataclass(frozen=True)
class EnrichmentDelta:
    """The cloud enrichment judge's parsed delta-envelope output.

    Exactly what :func:`_parse_enrichment_delta` already computed as a
    bare 4-tuple, named so :func:`_apply_enrichment_delta` and its callers
    (:func:`~paramem.graph.stage_enrich._stage_enrich`,
    :func:`~paramem.server.calibrate.calibrate_enrich`) can pass it around
    as one value instead of four positional ones.

    Attributes:
        add: New fact dicts to append (each already restricted to
            :data:`_FACT_FIELDS`).
        modify: ``(index, fields)`` pairs — a partial update for the
            indexed input fact (``fields`` already restricted to
            :data:`_FACT_FIELDS`).
        drop: Zero-based indices to remove from the input.
        bindings: New braced placeholders cloud introduced (key without
            braces, e.g. ``"Event_1"``) mapped to the exact
            anonymized-transcript span they stand for.
    """

    add: list[dict]
    modify: list[tuple[int, dict]]
    drop: set[int]
    bindings: dict[str, str]


def _parse_enrichment_delta(raw: str | None, n_facts: int) -> EnrichmentDelta | None:
    """Parse the cloud enrichment judge's delta-envelope output.

    Returns an :class:`EnrichmentDelta` on success; ``None`` when every
    retry attempt inside :func:`request_enrichment` either got no
    response at all or never returned the delta-envelope shape — the two
    causes :func:`request_enrichment`'s ``info["parse_path"]`` distinguishes
    (``"no_response"`` vs. ``"failed"``). The ``cloud_enrich`` call site
    (:func:`~paramem.graph.stage_enrich._stage_enrich`) branches on that
    distinction, NOT on this function's return value alone: ``"no_response"``
    (the provider was unreachable on every attempt) raises
    :class:`ExtractionFailed` so the batch aborts and its sessions stay
    pending for a clean retry; ``"failed"`` (the provider answered but never
    in the expected shape) fails OPEN — it keeps the pre-enrichment facts,
    records a ``cloud_enrichment_degraded`` diagnostic, and the cycle
    completes. A parse failure does NOT unconditionally abort the cycle.

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

    return EnrichmentDelta(add=add, modify=modify, drop=drop, bindings=bindings)


def _reconstruct_updated_transcript(
    anon_transcript: str | None,
    bindings: dict[str, str],
) -> str | None:
    """Substitute cloud-introduced bindings into the anonymized transcript.

    Replaces each binding's span with ``{{<placeholder>}}``, via
    :func:`~paramem.cloud.placeholders._substitute_whole_words` — longest
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
    # Single-brace `{Prefix_N}` matches the convention cloud used to echo
    # in the previous protocol's `updated_transcript` (saved snapshots
    # under `data/ha/debug/`) and the literal that `_apply_bindings`
    # already substitutes in fact subject / object.
    span_to_braced = {span: braced(placeholder) for placeholder, span in bindings.items()}
    return _substitute_whole_words(anon_transcript, span_to_braced)


def _apply_enrichment_delta(
    facts: list[dict],
    delta: EnrichmentDelta,
    scope: CloudScope | None,
    anon_transcript: str | None = None,
) -> tuple[list[dict], str | None, dict]:
    """Apply a parsed enrichment delta to input facts — per-triple
    accept/drop/revert, then reconstruct the updated transcript from the
    surviving bindings.

    Returns ``(facts, updated_transcript, report)``.  Unlike the retired
    whole-delta gate, this function ALWAYS returns a fact list — there is
    no parse-failure branch here: ``delta`` is already a parsed
    :class:`EnrichmentDelta`, and a parse failure (``delta=None`` from
    :func:`request_enrichment`) is handled one level up, BEFORE this
    function is ever called, by the caller — the ``enrich`` stage,
    :func:`~paramem.graph.stage_enrich._stage_enrich`. That caller does
    NOT unconditionally raise on ``None``: it branches on
    ``info["parse_path"]`` — ``"no_response"`` (provider unreachable on
    every attempt) raises :class:`ExtractionFailed` so the batch aborts;
    ``"failed"`` (provider answered but never in the expected shape) fails
    OPEN, keeping the pre-enrichment facts and recording a
    ``cloud_enrichment_degraded`` diagnostic — this function is not called
    either way, since both branches already know their fact list without
    an ``EnrichmentDelta`` to apply.

    ``scope`` supplies the resolvability domain — ``set(scope.resolution)``
    when given.  ``scope is None`` is a deliberate sentinel (the
    calibration / unit-test shape, mirroring
    :func:`~paramem.cloud.placeholders._apply_bindings`'s own
    ``observed=None`` convention): NOTHING is gated and every action
    applies exactly as written, since there is no
    :class:`~paramem.cloud.anonymize.AnonymizedContract` to resolve
    against.

    Per-action rules (owner-decided 2026-07-21/22 cloud-admission
    redesign — replaces the whole-delta totality gate that used to live in
    :func:`~paramem.cloud.deanonymize.deanonymize_facts`):

    1. ``add`` — a fact carrying any orphan token
       (:func:`~paramem.cloud.placeholders._fact_orphans`, scanning
       ``subject``/``object``) is DROPPED.  No loss: it never existed
       locally.
    2. ``modify`` — ``fields`` is shallow-merged into a COPY of the
       indexed input fact; if the RESULT carries any orphan, the fields
       are DISCARDED and the original input fact is kept UNCHANGED — the
       unit of rejection is cloud's CHANGE, not the fact itself.
    3. ``drop`` — honored UNCONDITIONALLY.  The spec's alternative
       (revert every drop in a delta that also had an add/modify
       rejection, preferring redundancy over loss) is explicitly NOT
       implemented here — the owner chose to measure the co-occurrence
       first (``report["drop_with_rejection"]``) before building that
       safety net.

    Application order mirrors the original: ``modify``, then ``drop``,
    then ``add``.

    ``updated_transcript`` excludes a binding ONLY when it is referenced
    EXCLUSIVELY by rejected content — an ``add``/``modify`` binding whose
    only referencing fact was rejected must not leak into the transcript.
    A binding referenced by nothing at all (never tied to any fact — a
    legal, if unusual, cloud mint) is NOT excluded; a binding referenced by
    both a rejected candidate and something that survived is NOT excluded
    either. Computed by scanning every rejected ``add``/``modify``
    candidate's tokens against the final surviving fact list's tokens
    AFTER every accept/drop/revert decision is made.

    ``report`` carries:

    * ``add_count`` / ``modify_count`` / ``drop_count`` / ``bindings_count``
      — the RAW counts from the parsed delta (same names
      :func:`request_enrichment` used to report before this split — kept
      for ``graph.diagnostics`` and ``tests/server/test_calibrate.py``
      consumers), before any rejection.
    * ``rejected_adds`` — count of ``add`` entries dropped for carrying an
      orphan.
    * ``reverted_modifies`` — count of ``modify`` entries whose fields
      were discarded for producing an orphan.
    * ``rejected_tokens`` — sorted list of the distinct orphan tokens
      found across every rejected ``add``/``modify``.
    * ``drop_with_rejection`` — ``True`` when this delta had a non-empty
      ``drop`` set AND at least one ``add``/``modify`` rejection — the
      measurement the owner asked for, to decide later whether reverting
      every drop on any rejection (spec rule 3, not implemented) is worth
      adding.
    """
    resolvable: set[str] | None = set(scope.resolution) if scope is not None else None

    working = [dict(f) for f in facts]
    rejected_tokens: set[str] = set()
    # Tokens referenced by a REJECTED add/modify candidate — used below to
    # scrub a binding from the transcript reconstruction ONLY when it is
    # referenced EXCLUSIVELY by rejected content (never by anything that
    # survived, and never left standalone/unreferenced-by-any-fact, which
    # is legitimate — a cloud mint need not be tied to a fact at all).
    rejected_reference_tokens: set[str] = set()
    reverted_modifies = 0
    for idx, fields in delta.modify:
        candidate = {**working[idx], **fields}
        orphans = _fact_orphans(candidate, resolvable) if resolvable is not None else set()
        if orphans:
            reverted_modifies += 1
            rejected_tokens |= orphans
            rejected_reference_tokens |= _fact_tokens(candidate)
            continue
        working[idx] = candidate

    surviving = [f for i, f in enumerate(working) if i not in delta.drop]

    rejected_adds = 0
    for fact in delta.add:
        orphans = _fact_orphans(fact, resolvable) if resolvable is not None else set()
        if orphans:
            rejected_adds += 1
            rejected_tokens |= orphans
            rejected_reference_tokens |= _fact_tokens(fact)
            continue
        surviving.append(fact)

    surviving_tokens: set[str] = set()
    for f in surviving:
        surviving_tokens |= _fact_tokens(f)
    rejected_only_tokens = rejected_reference_tokens - surviving_tokens
    surviving_bindings = {k: v for k, v in delta.bindings.items() if k not in rejected_only_tokens}

    report = {
        "add_count": len(delta.add),
        "modify_count": len(delta.modify),
        "drop_count": len(delta.drop),
        "bindings_count": len(delta.bindings),
        "rejected_adds": rejected_adds,
        "reverted_modifies": reverted_modifies,
        "rejected_tokens": sorted(rejected_tokens),
        "drop_with_rejection": bool(delta.drop) and bool(rejected_adds or reverted_modifies),
    }
    return (
        surviving,
        _reconstruct_updated_transcript(anon_transcript, surviving_bindings),
        report,
    )


def request_plausibility(
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
    """Cloud plausibility filter — drops invalid relations only.

    No additions, no modifications. See cloud_plausibility.txt for the
    drop criteria (self-loops, tautologies, role leaks, etc.).

    The judge emits a small ``{"drop": [<index>, ...]}`` object; this
    helper applies the drop-set to the input facts and returns the
    survivors.  Output is bounded and tiny by construction, so the
    truncation failure mode that hit the previous "echo every fact"
    protocol cannot recur on long inputs.

    Returns `(facts, raw_response)`. Raw response is preserved so callers
    can inspect the judge's verdict when questioning drop decisions.
    ``raw_response`` is ``None`` on a network/HTTP-level failure (the
    cloud call itself never returned) — the explicit branch below
    short-circuits on that case rather than relying on
    :func:`_apply_drop_set`/:func:`_parse_drop_set`'s own ``raw is None``
    handling to produce the same ``None`` result implicitly, matching
    :func:`request_enrichment`'s explicit ``raw is None`` branch (defect
    fixed 2026-07-21: this function used to fall through to
    ``_apply_drop_set`` unconditionally, the one asymmetry between the
    two sibling request functions).

    The prompt is external config — edit ``configs/prompts/cloud_plausibility.txt``
    to tune; no code changes are needed.

    ``prompts_dir`` overrides the search directory (forwarded to
    :func:`_load_prompt`) so a calibration override actually reaches the
    judge; it defaults to the production template.
    """
    plaus_prompt = _load_prompt("cloud_plausibility.txt", prompts_dir=prompts_dir, required=True)
    system_prompt = _load_prompt("cloud_plausibility_system.txt", required=True)
    facts_json, transcript_text = _cloud_facing_payload(enriched_anon_facts, anon_transcript)
    prompt = plaus_prompt.format(facts_json=facts_json, transcript=transcript_text)
    raw = _cloud_call(
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
        return None, None
    return _apply_drop_set(enriched_anon_facts, raw), raw


def judge_plausibility(
    facts: list[dict],
    transcript: str,
    model,
    tokenizer,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "cloud_plausibility.txt",
) -> tuple[list[dict] | None, str]:
    """Local-model plausibility filter — drops invalid relations only.

    Same prompt as the cloud plausibility filter, executed by a local model.
    Caller decides what data to pass: anonymized facts (placeholder strings)
    or de-anonymized facts (real names). The prompt is stage-agnostic.

    Returns ``(filtered_list, raw_output)``.  ``filtered_list`` is ``None``
    on parse failure (caller falls back).  The raw model output is the
    second element so calibration can capture it via phase_trace without
    re-running the call; an empty string indicates no raw response was
    obtained.

    The prompt is external config — edit ``configs/prompts/cloud_plausibility.txt``
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
    system_prompt = _load_prompt("cloud_plausibility_system.txt", required=True)
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
    token_count = estimate_tokens(formatted, tokenizer)
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
    cloud: dict | None = None,
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

    * ``cloud is None`` → LOCAL: ``model`` and ``tokenizer`` are required.
      Gradient checkpointing is disabled once before the candidate loop and
      re-enabled in ``finally``.  Each call is wrapped in
      ``vram_scope("dedup")``.
    * ``cloud`` dict → CLOUD: ``_cloud_call`` receives the RAW rendered prompt
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
        model: Local inference model.  Required when ``cloud`` is ``None``.
        tokenizer: Local tokenizer.  Required when ``cloud`` is ``None``.
        cloud: Cloud backend configuration dict with keys ``api_key``,
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

    # This primitive owns its phase, the way the local extraction primitive
    # owns ``local_extract``: prompt provenance and every group's raw model
    # output are captured identically wherever it runs — the graph tier's
    # production pass and a calibration run alike.  The enclosing
    # ``extraction_trace`` scope belongs to the pass that calls this.
    with phase_trace("normalize") as t:
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
            t.set_outcome(
                "no_input",
                reason="no (subject, object) group carries two or more predicates",
            )
            return clusters_by_so, diagnostics

        local_mode = cloud is None
        # Predicate-normalization is structured extraction: the local path must run on the
        # base weights (adapter disabled) with the KV cache live (checkpointing off,
        # restored to entry state on exit).  The cloud path uses the cloud model and
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
                    raw = _cloud_call(
                        rendered,
                        api_key=cloud["api_key"],
                        provider=cloud["provider"],
                        filter_model=cloud["filter_model"],
                        endpoint=cloud["endpoint"],
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

        t.set_raw("\n---\n".join(diagnostics["raw_outputs"]))
        t.set_parsed({k: v for k, v in diagnostics.items() if k != "raw_outputs"})
    return clusters_by_so, diagnostics
