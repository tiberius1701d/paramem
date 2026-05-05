"""LLM-based knowledge graph extraction — generate once, parse once."""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from paramem.graph.schema import Entity, SessionGraph
from paramem.graph.schema_config import (
    anonymizer_placeholder_pattern,
    anonymizer_prefix_to_type,
    anonymizer_type_to_prefix,
    fallback_entity_type,
    fallback_relation_type,
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
    format_replacement_rules,
    relation_types,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"

_DEFAULT_EXTRACTION_SYSTEM = "You are a precise knowledge graph extractor. Output valid JSON only."

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
_DEFAULT_FILTER_MAX_TOKENS = 8192
# Validator temperature: deterministic by default. Threaded all the way to the
# provider call so Anthropic and OpenAI-compatible filters match exactly.
_DEFAULT_FILTER_TEMPERATURE = 0.0
# Per-call timeout for SOTA enrichment / plausibility. 30s was hit by Mistral 7B
# resume content at max_tokens=8192 (response generation took >30s, ReadTimeout,
# pipeline fell back to local-only and lost SOTA's contribution silently). 90s
# matches CloudAgentConfig.timeout_seconds default. Operator can override via
# the timeout_seconds parameter on the relevant call site.
_DEFAULT_FILTER_TIMEOUT_SECONDS = 90.0


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


# ---------------------------------------------------------------------------
# Word-boundary substitution / matching helpers — replace fragile
# ``re.sub(rf"\b{re.escape(name)}\b", ...)`` patterns on user-content text
# with structural token walks.
# ---------------------------------------------------------------------------


def _is_word_char(c: str) -> bool:
    """Match Python regex ``\\w`` semantics: alphanumeric (Unicode-aware) or underscore."""
    return c.isalnum() or c == "_"


def _substitute_whole_words(
    text: str,
    mapping: dict[str, str],
    *,
    case_insensitive: bool = False,
) -> str:
    """Replace whole-word occurrences of mapping keys with their values.

    Mirrors the previous ``for k, v in mapping: text = re.sub(rf"\\b{re.escape(k)}\\b", v, text)``
    pattern in a single token-walk pass.  Boundaries follow the ``\\b``
    semantics: word-character/non-word-character transitions, where word-
    character means ``c.isalnum() or c == "_"`` (Unicode-aware).

    Longest keys are tried first at each position so multi-word keys
    preempt single-word prefixes (``"Person_2"`` before ``"Person"``).
    Empty / non-string keys are skipped defensively — local extractors
    occasionally emit ``null`` mapping entries.
    """
    if not mapping or not text:
        return text
    if case_insensitive:
        normalized = {k.lower(): v for k, v in mapping.items() if isinstance(k, str)}
    else:
        normalized = {k: v for k, v in mapping.items() if isinstance(k, str)}
    keys_sorted = sorted((k for k in normalized if k), key=len, reverse=True)
    if not keys_sorted:
        return text

    parts: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        if not _is_word_char(text[pos]):
            parts.append(text[pos])
            pos += 1
            continue
        matched = False
        for key in keys_sorted:
            klen = len(key)
            end = pos + klen
            if end > n:
                continue
            slice_ = text[pos:end]
            if case_insensitive:
                if slice_.lower() != key:
                    continue
            elif slice_ != key:
                continue
            if end < n and _is_word_char(text[end]):
                continue
            replacement = normalized[key]
            if not isinstance(replacement, str):
                continue
            parts.append(replacement)
            pos = end
            matched = True
            break
        if not matched:
            j = pos + 1
            while j < n and _is_word_char(text[j]):
                j += 1
            parts.append(text[pos:j])
            pos = j
    return "".join(parts)


def _contains_whole_word(text: str, word: str, *, case_insensitive: bool = False) -> bool:
    """True iff ``word`` appears as a whole word in ``text``.

    Mirrors ``bool(re.search(rf"\\b{re.escape(word)}\\b", text))`` (with
    ``re.IGNORECASE`` when ``case_insensitive=True``).  Same boundary
    definition as :func:`_substitute_whole_words`.
    """
    if not word or not text or len(word) > len(text):
        return False
    haystack = text.lower() if case_insensitive else text
    needle = word.lower() if case_insensitive else word
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx < 0:
            return False
        if idx > 0 and _is_word_char(haystack[idx - 1]):
            pos = idx + 1
            continue
        end = idx + len(needle)
        if end < len(haystack) and _is_word_char(haystack[end]):
            pos = idx + 1
            continue
        return True


def _extract_alpha_tokens(text: str, *, min_len: int = 1) -> list[str]:
    """Split text into runs of alphabetic characters.

    Mirrors ``re.findall(r"[a-z]+", text)`` semantics on lowercased input.
    ASCII-only matches the previous behaviour — Unicode handling is a
    separate decision the original code didn't take.
    """
    tokens: list[str] = []
    current: list[str] = []
    for c in text:
        if "a" <= c <= "z":
            current.append(c)
            continue
        if current:
            tok = "".join(current)
            if len(tok) >= min_len:
                tokens.append(tok)
            current = []
    if current:
        tok = "".join(current)
        if len(tok) >= min_len:
            tokens.append(tok)
    return tokens


_NER_APOSTROPHES = ("'", "’")


def _strip_ner_dialogue_tail(text: str) -> str:
    """Strip a ``:<whitespace><word-char>...$`` dialogue tail.

    Mirrors ``re.sub(r":\\s*\\w+.*$", "", text)``: a colon followed by
    optional whitespace, then a word character, then anything to end of
    string is removed (along with the colon itself).  Returns ``text``
    unchanged when no such suffix is present.
    """
    if ":" not in text:
        return text
    pos = 0
    while True:
        colon = text.find(":", pos)
        if colon < 0:
            return text
        scan = colon + 1
        while scan < len(text) and text[scan].isspace():
            scan += 1
        if scan < len(text) and _is_word_char(text[scan]):
            return text[:colon]
        pos = colon + 1


def _strip_ner_possessive(text: str) -> str:
    """Strip a trailing possessive (``'`` / ``'s`` / ``’`` / ``’s``).

    Mirrors ``re.sub(r"['’]s?$", "", text)``.
    """
    if not text:
        return text
    if text.endswith("s") and len(text) > 1 and text[-2] in _NER_APOSTROPHES:
        return text[:-2]
    if text[-1] in _NER_APOSTROPHES:
        return text[:-1]
    return text


# Prompt filename constants — one definition site; imported by consolidation.py.
DEFAULT_SYSTEM_PROMPT_FILENAME = "extraction_system.txt"
DEFAULT_USER_PROMPT_FILENAME = "extraction.txt"
DOCUMENT_SYSTEM_PROMPT_FILENAME = "extraction_system_document.txt"
DOCUMENT_USER_PROMPT_FILENAME = "extraction_document.txt"
DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME = "extraction_procedural.txt"
DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME = "extraction_procedural_document.txt"


def build_speaker_context(speaker_name: str | None) -> str:
    """Single source of truth for the extraction-prompt speaker directive.

    Empty string when the speaker cannot be identified, leaving the
    ``{SPEAKER_NAME}`` slot in the few-shots unsubstituted (the prompt's
    closing note tells the model never to emit that literal string).
    When known, pins the real name — which may be a real first name
    ("Alice") or an opaque anonymous id ("Speaker7") — as the canonical
    subject across every extracted fact.
    """
    if not speaker_name:
        return ""
    return (
        f"\nThe current speaker is {speaker_name}. Use the exact string "
        f"'{speaker_name}' as the subject of every fact about the speaker; "
        f"do NOT use '{{SPEAKER_NAME}}', 'SPEAKER_NAME', 'Speaker_Name', "
        f"'Speaker', 'User', 'I', or any other placeholder.\n"
    )


_DEFAULT_EXTRACTION_PROMPT = """\
Extract all entities and relations from this conversation transcript.{speaker_context}

Extract as JSON with `entities` and `relations` arrays.

Transcript:
{transcript}
"""


def _load_prompt(filename: str, default: str, prompts_dir: Path | None = None) -> str:
    """Load a prompt file, falling back to hardcoded default."""
    search_dirs = []
    if prompts_dir:
        search_dirs.append(Path(prompts_dir))
    search_dirs.append(_DEFAULT_PROMPT_DIR)

    for d in search_dirs:
        path = d / filename
        if path.exists():
            return path.read_text().strip()
    return default


def load_extraction_prompts(
    prompts_dir: str | Path | None = None,
    *,
    system_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_filename: str = DEFAULT_USER_PROMPT_FILENAME,
) -> tuple[str, str]:
    """Load extraction prompts from a directory, with hardcoded fallbacks.

    Args:
        prompts_dir: Directory containing the prompt files.  Falls back to
                     ``configs/prompts/`` in the project root, then to
                     hardcoded defaults.
        system_filename: Filename of the system prompt.  Defaults to
                         :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
                         (``"extraction_system.txt"``); pass
                         :data:`DOCUMENT_SYSTEM_PROMPT_FILENAME`
                         (``"extraction_system_document.txt"``) for the
                         written-document extraction variant.
        user_filename: Filename of the user-turn prompt template.  Defaults to
                       :data:`DEFAULT_USER_PROMPT_FILENAME`
                       (``"extraction.txt"``); pass
                       :data:`DOCUMENT_USER_PROMPT_FILENAME`
                       (``"extraction_document.txt"``) for the document variant.

    Returns:
        ``(system_prompt, extraction_prompt)`` tuple.
    """
    pd = Path(prompts_dir) if prompts_dir else None
    system = _load_prompt(system_filename, _DEFAULT_EXTRACTION_SYSTEM, pd)
    prompt = _load_prompt(user_filename, _DEFAULT_EXTRACTION_PROMPT, pd)
    return system, prompt


_DEFAULT_PROCEDURAL_PROMPT = """\
Extract preferences, habits, and routines from this conversation.{speaker_context}
Only extract relation_type "preference". Return JSON.

Transcript:
{transcript}

Return JSON with entities, relations, summary.
"""

# _DEFAULT_EXTRACTION_PROMPT and _DEFAULT_PROCEDURAL_PROMPT intentionally do
# NOT include {entity_types} or {predicate_examples} placeholders — they are
# self-contained fallbacks used only when the prompt files cannot be read.
# The file-based prompts (extraction.txt, extraction_procedural.txt) carry
# those placeholders and receive the formatted values via .format() kwargs.


def load_procedural_prompt(
    prompts_dir: str | Path | None = None,
    *,
    system_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_filename: str = DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
) -> tuple[str, str]:
    """Load procedural extraction prompts.

    Args:
        prompts_dir: Directory containing the prompt files.  Falls back to
                     ``configs/prompts/`` in the project root.
        system_filename: Filename of the system prompt.  Defaults to
                         :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`; pass
                         :data:`DOCUMENT_SYSTEM_PROMPT_FILENAME` when
                         extracting procedural facts from a written document
                         so the model is not primed with dialogue-style
                         few-shots.
        user_filename: Filename of the user-turn prompt template.  Defaults to
                       :data:`DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME`
                       (``"extraction_procedural.txt"``); pass
                       :data:`DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME`
                       (``"extraction_procedural_document.txt"``) for written
                       documents so the model is not primed with
                       dialogue-shaped few-shots that reference a non-existent
                       assistant response.
    """
    pd = Path(prompts_dir) if prompts_dir else None
    system = _load_prompt(system_filename, _DEFAULT_EXTRACTION_SYSTEM, pd)
    prompt = _load_prompt(user_filename, _DEFAULT_PROCEDURAL_PROMPT, pd)
    return system, prompt


def extract_procedural_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    speaker_id: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    prompts_dir: str | Path | None = None,
    stt_correction: bool = True,
    speaker_name: str | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
) -> SessionGraph:
    """Extract preferences/habits from a session transcript.

    Separate extraction pass with a dedicated prompt targeting
    behavioral patterns rather than factual knowledge.

    Args:
        speaker_name: Real name of the speaker (e.g. from voice enrollment).
            When provided, injected into the prompt via ``build_speaker_context``
            so the model uses the real name as the subject of every extracted
            preference instead of the ``SPEAKER_NAME`` slot. Mirrors
            the same parameter on ``extract_graph``.
        speaker_id: Speaker store ID (e.g. ``"Speaker0"``). Stamped onto every
            ``Relation`` extracted in this pass as provenance. Required —
            callers must always supply a real speaker ID.
        stt_correction: Correct entity names from the assistant response turn.
            This is a no-op when
            ``user_prompt_filename=DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME``
            because the document path has no assistant response to
            cross-reference; passing ``stt_correction=True`` with a document
            source is harmless but produces no correction.
        system_prompt_filename: Filename of the system prompt within the prompts
            directory.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
            (dialogue variant); pass :data:`DOCUMENT_SYSTEM_PROMPT_FILENAME`
            when the source is a written document so the model is not
            instructed to cross-reference a non-existent assistant turn.
        user_prompt_filename: Filename of the user-turn prompt template.
            Defaults to :data:`DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME`
            (``"extraction_procedural.txt"``); pass
            :data:`DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME`
            (``"extraction_procedural_document.txt"``) for written-document
            sources so the model receives document-shaped few-shots instead of
            dialogue-shaped ones.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_procedural_prompt(
        prompts_dir,
        system_filename=system_prompt_filename,
        user_filename=user_prompt_filename,
    )
    speaker_context = build_speaker_context(speaker_name)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": prompt.format(
                transcript=transcript,
                speaker_context=speaker_context,
                entity_types=format_entity_types(scope="procedural"),
                predicate_examples=format_predicate_examples(scope="procedural"),
            ),
        },
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )

    raw_output = generate_answer(
        model, tokenizer, formatted, max_new_tokens=max_tokens, temperature=temperature
    )
    logger.debug("Procedural extraction raw: %s", raw_output[:500])

    try:
        json_str = _extract_json_block(raw_output)
        data = json.loads(json_str)
        data["session_id"] = session_id
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data = _normalize_extraction(data)
        # Stamp speaker_id onto every relation dict before schema validation.
        # Relation.speaker_id is mandatory; the LLM output never includes it.
        for rel_dict in data.get("relations", []):
            rel_dict.setdefault("speaker_id", speaker_id)
        graph = SessionGraph.model_validate(data)
        # Bind speaker Entity and fold any split-name duplicates.
        if speaker_name:
            graph = _stamp_speaker_entity(graph, speaker_name=speaker_name, speaker_id=speaker_id)
    except Exception as exc:
        logger.warning("Procedural extraction failed (%s), returning empty", exc)
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if graph.relations and stt_correction:
        graph = _correct_entity_names(graph, transcript)

    logger.info(
        "Procedural extraction: %d entities, %d relations (session=%s)",
        len(graph.entities),
        len(graph.relations),
        session_id,
    )
    return graph


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
    ha_context: dict | None = None,
    stt_correction: bool = True,
    ha_validation: bool = True,
    noise_filter: str = "",
    noise_filter_model: str = "claude-sonnet-4-6",
    noise_filter_endpoint: str | None = None,
    speaker_name: str | None = None,
    ner_check: bool = False,
    ner_model: str = "en_core_web_sm",
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    verify_anonymization: bool = True,
    role_aware_grounding: str = "off",
    pii_scope: set[str] | frozenset[str] | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Multi-pass pipeline:
    1. Extract candidate triples from transcript
    2. Correct STT entity names from assistant responses (configurable)
    3. Validate with HA context — location ground truth (configurable)
    4. SOTA pipeline (anonymize → enrich → plausibility → de-anonymize, configurable)

    All filters fail gracefully — extraction result is preserved on any failure.

    Args:
        temperature: Sampling temperature for extraction (default 0.0 for determinism).
        max_tokens: Max output tokens for extraction (default 2048).
        prompts_dir: Optional override for prompt config directory.
        validate: Run SOTA pipeline pass 4 (default True). Passes 2-3 have
            their own flags (stt_correction, ha_validation).
        ha_context: HA home config for location validation (from get_home_context).
        stt_correction: Correct entity names from assistant responses.
            This is a no-op when
            ``user_prompt_filename=DOCUMENT_USER_PROMPT_FILENAME`` because the
            document path has no assistant response to cross-reference; passing
            ``stt_correction=True`` with a document source is harmless but
            produces no correction.
        ha_validation: Validate locations against HA home context.
        noise_filter: SOTA provider for noise filtering ("" = disabled).
        ner_check: Enable spaCy NER cross-check for PII detection (default False).
        ner_model: spaCy model for NER when ner_check=True.
        plausibility_judge: Plausibility filter judge ("auto"=local, "off"=disabled,
            or a SOTA provider name like "claude" for cloud judging at anon stage).
        plausibility_stage: When to run plausibility ("deanon"=after de-anon,
            "anon"=on anonymized data with SOTA judge).
        verify_anonymization: Run forward-path privacy guard before SOTA (default True).
        speaker_id: Speaker store ID (e.g. ``"Speaker0"``). Stamped onto every
            ``Relation`` produced by this extraction pass as provenance.
            Required — callers must always supply the session's speaker ID.
        system_prompt_filename: Filename of the system prompt within the prompts
            directory.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
            (``"extraction_system.txt"``); pass
            :data:`DOCUMENT_SYSTEM_PROMPT_FILENAME`
            (``"extraction_system_document.txt"``) for written-document sources.
        user_prompt_filename: Filename of the user-turn prompt template.  Defaults
            to :data:`DEFAULT_USER_PROMPT_FILENAME` (``"extraction.txt"``); pass
            :data:`DOCUMENT_USER_PROMPT_FILENAME`
            (``"extraction_document.txt"``) for written-document sources.
    """
    raw_output = _generate_extraction(
        model,
        tokenizer,
        transcript,
        temperature,
        max_tokens,
        prompts_dir,
        speaker_name,
        system_prompt_filename=system_prompt_filename,
        user_prompt_filename=user_prompt_filename,
    )
    logger.debug("Raw extraction output: %s", raw_output[:500])

    try:
        graph = _parse_extraction(
            raw_output,
            session_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )
    except Exception as exc:
        logger.warning(
            "Extraction parsing failed (%s), returning empty graph",
            exc,
        )
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if not graph.relations:
        return graph

    # Pass 2: correct STT entity names from assistant responses
    if stt_correction:
        graph = _correct_entity_names(graph, transcript)

    # Pass 3: validate location facts against HA home context
    if ha_validation and ha_context:
        graph = _validate_with_ha_context(graph, ha_context)

    # Pass 4: SOTA pipeline (anonymize → enrich → plausibility filter → de-anonymize)
    if validate and noise_filter and graph.relations:
        graph = _sota_pipeline(
            graph,
            transcript,
            model,
            tokenizer,
            provider=noise_filter,
            filter_model=noise_filter_model,
            endpoint=noise_filter_endpoint,
            ner_check=ner_check,
            ner_model=ner_model,
            plausibility_judge=plausibility_judge,
            plausibility_stage=plausibility_stage,
            verify_anonymization=verify_anonymization,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            role_aware_grounding=role_aware_grounding,
            pii_scope=pii_scope,
            max_tokens=max_tokens,
            plausibility_max_tokens=plausibility_max_tokens,
        )

    return graph


def extract_and_anonymize_for_cloud(
    transcript: str,
    model,
    tokenizer,
    *,
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    pii_scope: set[str] | frozenset[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Local extract + local anonymize for cloud egress.

    Composition over existing primitives — same anonymization sequence
    ``_sota_pipeline`` runs every consolidation cycle, minus the SOTA
    enrichment call:

    1. ``extract_graph(validate=False)`` — local extraction only,
       produces a SessionGraph the anonymizer can anchor on.
    2. ``_anonymize_with_local_model(graph, transcript=transcript)`` —
       model-based anonymization of facts + transcript.
    3. Mechanical transcript fallback when the model omits
       ``anonymized_transcript`` (older prompt schema returns facts only).
    4. ``_normalize_anonymization_mapping`` — canonicalize direction.
    5. ``verify_anonymization_completeness`` + ``_repair_anonymization_leaks``
       — extend mapping for missed names, drop triples for hallucinated
       ones (canonical-mapping path only).
    6. Final completeness check; any residual leak → block.

    ``pii_scope`` controls which NER categories are anonymized; passes
    through to NER and verify.  ``None`` → :data:`_CLOUD_EGRESS_DEFAULT_SCOPE`
    (``{"person"}``) — narrower than the primitive default because the
    cloud-utility tradeoff (Berlin restaurants, organisation-aware
    advice) bites here.  An empty scope short-circuits before any LLM
    call: the helper returns ``(transcript, {})`` and the caller sends
    the original text to the cloud verbatim — no anonymization, no
    deanonymization needed.

    Return shapes:

    * ``(anon_transcript, mapping)`` — anonymization ran, mapping is
      non-empty.  Caller deanonymizes the cloud's response with
      :func:`deanonymize_text`.
    * ``(transcript, {})`` — operator opted out (``pii_scope=[]``) or
      the input had no in-scope content; caller forwards verbatim.
    * ``("", {})`` — block.  Extraction error, anonymizer parse
      failure, residual leak after repair, or non-canonical mapping.
      Caller skips the cloud call.

    The companion :func:`deanonymize_text` is a no-op on an empty
    mapping, so callers can apply it unconditionally.
    """
    if not transcript or not transcript.strip():
        return "", {}

    scope = _CLOUD_EGRESS_DEFAULT_SCOPE if pii_scope is None else frozenset(pii_scope)
    # Empty scope = operator opt-out.  Skip the entire LLM-driven
    # anonymization path and let the caller forward the transcript
    # verbatim.  Distinguished from ``("", {})`` block by non-empty text.
    if not scope:
        return transcript, {}

    try:
        graph = extract_graph(
            model,
            tokenizer,
            transcript,
            session_id="cloud_egress",
            speaker_name=speaker_name,
            prompts_dir=prompts_dir,
            validate=False,
            stt_correction=False,
            ha_validation=False,
            noise_filter="",
        )
    except Exception:
        logger.exception("Cloud egress: local extraction failed; treating as block")
        return "", {}

    if not graph.relations:
        return "", {}

    try:
        anon_facts, mapping, anon_transcript = _anonymize_with_local_model(
            graph, model, tokenizer, transcript=transcript
        )
    except Exception:
        logger.exception("Cloud egress: anonymization raised; treating as block")
        return "", {}

    if anon_facts is None or not mapping:
        return "", {}

    mapping, _norm_stats = _normalize_anonymization_mapping(mapping)
    if not anon_transcript:
        # Older anonymization prompt returns only facts; rebuild the
        # transcript mechanically from the mapping (same fallback
        # `_sota_pipeline` uses).
        anon_transcript = _anonymize_transcript(transcript, mapping)

    # Defense-in-depth: NER cross-check catches PII that extraction missed
    # (e.g. Mistral 7B emits relations referencing place names without
    # tagging them as entities — those slip past the entity-scoped verify).
    # Cloud egress is privacy-critical so we always enable NER cross-check;
    # falls back to no-op if spaCy isn't installed.  ``pii_scope`` filters
    # which categories NER surfaces.
    extra_pii = extract_pii_names_with_ner(transcript, pii_scope=scope)

    leaked = verify_anonymization_completeness(
        graph,
        mapping,
        anon_facts,
        anon_transcript,
        extra_pii_names=extra_pii,
        pii_scope=scope,
    )
    if leaked:
        if not _mapping_is_canonical(mapping):
            logger.warning(
                "Cloud egress: residual leaks with non-canonical mapping (%s); blocking",
                leaked[:3],
            )
            return "", {}
        anon_facts, mapping, anon_transcript, _status = _repair_anonymization_leaks(
            graph,
            mapping,
            anon_facts,
            anon_transcript,
            transcript,
            leaked,
            extra_pii_types=extra_pii,
        )
        leaked = verify_anonymization_completeness(
            graph,
            mapping,
            anon_facts,
            anon_transcript,
            extra_pii_names=extra_pii,
            pii_scope=scope,
        )
        if leaked:
            logger.warning("Cloud egress: residual leaks after repair (%s); blocking", leaked[:3])
            return "", {}

    if not anon_transcript or not mapping:
        return "", {}

    return anon_transcript, mapping


def deanonymize_text(text: str, mapping: dict[str, str]) -> str:
    """Restore real names in cloud-returned text via the reverse mapping.

    ``mapping`` is the forward direction (``real -> placeholder``);
    this function inverts it internally.  Word-boundary anchored, so
    a placeholder embedded in unrelated text doesn't match.
    Idempotent on text without placeholders or with empty mapping.
    """
    if not text or not mapping:
        return text
    reverse = {v: k for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
    return _substitute_whole_words(text, reverse)


def _generate_extraction(
    model,
    tokenizer,
    transcript: str,
    temperature: float,
    max_tokens: int,
    prompts_dir: str | Path | None = None,
    speaker_name: str | None = None,
    *,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
) -> str:
    """Generate graph extraction output from the model. Called once.

    When ``speaker_name`` is provided (e.g. from voice enrollment in
    production, or from session metadata in the test harness), inject it
    into the prompt so the model uses the real name as subject instead of
    guessing or emitting the ``SPEAKER_NAME`` slot from the few-shots.

    The system prompt is passed verbatim — no slot substitution is performed
    on it.  Narrator binding is achieved via the ``{speaker_context}``
    placeholder in the **user** template (both ``extraction.txt`` and
    ``extraction_document.txt`` carry this slot), populated by
    :func:`build_speaker_context`.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_extraction_prompts(
        prompts_dir,
        system_filename=system_prompt_filename,
        user_filename=user_prompt_filename,
    )
    speaker_context = build_speaker_context(speaker_name)
    format_kwargs = dict(
        transcript=transcript,
        speaker_context=speaker_context,
        entity_types=format_entity_types(),
        predicate_examples=format_predicate_examples(),
        relation_types=format_relation_types(),
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

    return generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


def _parse_extraction(
    raw_output: str,
    session_id: str,
    speaker_id: str,
    speaker_name: str | None = None,
) -> SessionGraph:
    """Parse raw model output into a SessionGraph.

    Handles non-standard field names, array-valued fields, and other
    model output quirks via _normalize_extraction. Local models occasionally
    emit a bare JSON array of fact dicts instead of the expected
    ``{"entities": [...], "relations": [...]}`` envelope; that case is
    rewrapped here so downstream normalization can proceed.

    After schema validation, :func:`_stamp_speaker_entity` is called to:
    - Stamp ``speaker_id`` on the entity whose name matches ``speaker_name``.
    - Fold any duplicate speaker-named entity (e.g. "Alex Morgan" alongside
      "Alex") into the canonical one and rewrite all affected relation
      subjects/objects.

    Args:
        raw_output: Raw model output string.
        session_id: Session identifier for the graph.
        speaker_id: Speaker store ID stamped onto every relation as provenance.
            Required — callers must always supply the session's speaker ID.
        speaker_name: Real display name of the speaker (e.g. "Alex" or
            "Speaker0").  When provided, the speaker Entity is identified and
            its ``speaker_id`` field is set.  When ``None`` the post-processing
            step is skipped.
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
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    data = _normalize_extraction(data)

    # Stamp speaker_id onto every relation dict before schema validation.
    # Relation.speaker_id is mandatory; the LLM output never includes it.
    for rel_dict in data.get("relations", []):
        rel_dict.setdefault("speaker_id", speaker_id)

    graph = SessionGraph.model_validate(data)

    # Post-process: bind speaker Entity and fold any split-name duplicates.
    if speaker_name:
        graph = _stamp_speaker_entity(graph, speaker_name=speaker_name, speaker_id=speaker_id)

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
    speaker_name: str,
    speaker_id: str,
) -> SessionGraph:
    """Stamp ``speaker_id`` on the canonical speaker Entity and fold duplicates.

    Two tasks are performed on the parsed :class:`~paramem.graph.schema.SessionGraph`:

    1. **Stamp** — find the entity whose ``name`` matches ``speaker_name``
       (case-insensitive) and set its ``speaker_id`` to ``speaker_id``.  If no
       exact match is found, the entity whose name starts with the
       ``speaker_name`` prefix is treated as the canonical entity (e.g. "Alex"
       found as prefix of "Alex Morgan" — only when the speaker's display name
       is the first-name part of a longer name the model emitted).

    2. **Fold** — if the model emitted a *second* entity whose name is a
       longer form of the speaker's name (e.g. speaker_name="Alex" and model
       also emitted "Alex Morgan"), fold the duplicate into the canonical
       entity: merge attributes, rewrite every relation whose subject or object
       matched the duplicate name to the canonical name, and remove the
       duplicate from the entity list.

    Any entity whose name matches ``speaker_id`` exactly (opaque IDs like
    "Speaker0") is also treated as a speaker alias and folded.

    This post-processor is a defensive measure against the model ignoring the
    prompt's "collapse self-references" instruction.  It is idempotent — if
    the model obeyed the prompt perfectly only task 1 fires (no duplicates to
    fold).

    Args:
        graph: Parsed :class:`~paramem.graph.schema.SessionGraph` after
            schema validation.
        speaker_name: Canonical display name of the speaker as passed to the
            extraction prompt (e.g. ``"Alex"`` or ``"Speaker0"``).
        speaker_id: Speaker store ID to stamp onto the canonical entity.

    Returns:
        Updated ``SessionGraph`` with ``speaker_id`` set on the canonical
        speaker Entity and any duplicate entities removed.
    """
    if not graph.entities:
        return graph

    speaker_name_lower = speaker_name.lower()
    speaker_id_lower = speaker_id.lower()

    # Identify the canonical entity: exact name match first, then prefix match.
    canonical: Entity | None = None
    for ent in graph.entities:
        if ent.name.lower() == speaker_name_lower:
            canonical = ent
            break

    if canonical is None:
        # Fallback: entity whose name starts with speaker_name (full-name variants).
        # Do NOT rename canonical.name to speaker_name here — relations still
        # reference the longer form, and the privacy verifier would then flag
        # the bare display name as a leak when it appears as a substring of
        # email/URL tokens (e.g. "alex" inside "alex.smith@example.com"
        # — the dot is a regex word boundary). The graph identity is
        # speaker_id (stamped below), not the name string; rendering layers
        # can resolve display names via SpeakerStore.get_name() at render time.
        for ent in graph.entities:
            if ent.name.lower().startswith(speaker_name_lower + " "):
                canonical = ent
                break

    if canonical is None:
        # No match at all — nothing to stamp or fold.
        return graph

    # Stamp speaker_id.
    canonical.speaker_id = speaker_id

    # Collect alias names: any entity that is a longer form of the speaker name
    # or matches the opaque speaker_id, excluding the canonical entity itself.
    alias_names: set[str] = set()
    for ent in graph.entities:
        if ent is canonical:
            continue
        name_lower = ent.name.lower()
        # Full-name variant (e.g. "Alex Morgan" when canonical is "Alex")
        if name_lower.startswith(speaker_name_lower + " ") or name_lower.startswith(
            speaker_name_lower + "-"
        ):
            alias_names.add(ent.name)
        # Opaque speaker_id entity (e.g. "Speaker0" entity alongside "Alex")
        elif name_lower == speaker_id_lower:
            alias_names.add(ent.name)

    if not alias_names:
        return graph

    # Merge attributes from all alias entities into canonical.
    alias_map: dict[str, str] = {}  # alias_name -> canonical.name
    entities_to_remove: set[str] = set()
    for ent in graph.entities:
        if ent.name in alias_names:
            canonical.attributes.update(ent.attributes)
            alias_map[ent.name] = canonical.name
            entities_to_remove.add(ent.name)
            logger.info(
                "Speaker entity fold: '%s' → '%s' (speaker_id=%s)",
                ent.name,
                canonical.name,
                speaker_id,
            )

    # Rewrite relation subjects/objects that referenced alias names.
    for rel in graph.relations:
        if rel.subject in alias_map:
            rel.subject = alias_map[rel.subject]
        if rel.object in alias_map:
            rel.object = alias_map[rel.object]

    # Remove folded entity objects.
    graph.entities = [e for e in graph.entities if e.name not in entities_to_remove]

    return graph


_JSON_ENVELOPE_KEYS = frozenset(
    (
        # Local extractor / SOTA enrichment / procedural envelopes
        "facts",
        "entities",
        "relations",
        "new_entity_bindings",
        "summary",
        # Anonymizer envelope — `{"anonymized": [...], "mapping": {...}}`
        "anonymized",
        "mapping",
    )
)


def _extract_json_block(text: str) -> str:
    """Extract a JSON object/array envelope from model output.

    Handles three real-world failure modes the model produces:

    1. **Code-fenced output**: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\``` — the
       fence is stripped before scanning.
    2. **Prose preamble with brace-quoted placeholder names**: SOTA
       sometimes narrates "I'll introduce ``{Topic_1}`` for the degree
       field" *before* emitting the JSON envelope.  The literal
       ``{Topic_1}`` is not parseable JSON (unquoted key) — a naive
       "first ``{``" parser would fail there and miss the real envelope
       further down.
    3. **Truncated outer envelope**: a model cut at ``max_tokens`` mid-
       string emits a valid *inner* sub-object ``{"name": "x", ...}``
       even though the outer envelope never closed.  Returning that
       inner object would silently produce an empty graph downstream.

    Algorithm: walk every ``{`` / ``[`` position in the text and try
    ``raw_decode`` from each.  Accept the first candidate that yields
    either a non-empty list, or a dict with at least one envelope key
    (``facts`` / ``entities`` / ``relations`` / ``new_entity_bindings``
    / ``summary``).  Inner sub-objects without envelope keys are
    skipped so truncation surfaces as a parse failure rather than a
    silently empty graph.  Preamble noise (``{Topic_1}``) is skipped
    automatically because it raises ``JSONDecodeError`` immediately.

    Error messages distinguish three fail modes:
    a. ``saw_decode_success_no_envelope=True`` → JSON parsed but no
       envelope keys → likely outer-envelope truncation.
    b. ``last_exc is not None`` → some ``{`` / ``[`` candidates existed
       but none parsed → likely all garbage / prose / corruption.
    c. No candidates at all → no JSON in the response.
    """
    # Strip markdown code-fence wrappers if present.
    src = text
    for marker in ("```json", "```"):
        if marker in src:
            start = src.index(marker) + len(marker)
            closing = src.find("```", start)
            if closing != -1:
                src = src[start:closing].strip()
                break

    decoder = json.JSONDecoder()
    last_exc: json.JSONDecodeError | None = None
    last_offset = -1
    saw_decode_success_no_envelope = False
    src_len = len(src)
    pos = 0
    while pos < src_len:
        next_brace = src.find("{", pos)
        next_bracket = src.find("[", pos)
        if next_brace < 0 and next_bracket < 0:
            break
        if next_brace < 0:
            candidate = next_bracket
        elif next_bracket < 0:
            candidate = next_brace
        else:
            candidate = min(next_brace, next_bracket)
        last_offset = candidate
        try:
            value, end = decoder.raw_decode(src, candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
            # Skip past the failing opener and keep looking.
            pos = candidate + 1
            continue
        # raw_decode succeeded — check whether this is the envelope or
        # narrative noise / a truncated inner sub-object.
        # Acceptance rules — both designed to surface truncation rather
        # than silently consume a sub-structure of a truncated envelope:
        #   • dict: must carry an envelope key (entities / relations /
        #     facts / new_entity_bindings / summary). Rejects naked
        #     entity / relation dicts that survive a truncated outer
        #     envelope.
        #   • list: must be empty (plausibility's "all dropped" output)
        #     OR have first element shaped like a fact (subject /
        #     predicate / object key present). Rejects an entity list
        #     that survives a truncated outer envelope — its elements
        #     have ``name`` / ``entity_type``, not ``subject``.
        if isinstance(value, dict):
            is_envelope = bool(set(value.keys()) & _JSON_ENVELOPE_KEYS)
        elif isinstance(value, list):
            if not value:
                is_envelope = True
            else:
                first = value[0]
                is_envelope = isinstance(first, dict) and (
                    "subject" in first or "predicate" in first or "object" in first
                )
        else:
            is_envelope = False
        if is_envelope:
            return src[candidate:end]
        saw_decode_success_no_envelope = True
        pos = end

    if saw_decode_success_no_envelope:
        # Some JSON parsed cleanly but none had envelope keys — most
        # likely the outer envelope was truncated at ``max_tokens`` and
        # only inner sub-objects survived.
        raise ValueError(
            "Parsed JSON values found in model output but none have "
            "envelope keys (entities/relations/facts/new_entity_bindings/"
            "summary; non-empty list also accepted). Likely cause: outer "
            f"envelope truncated at max_tokens (response length {src_len}). "
            "Bump extraction_max_tokens or reduce input chunk size."
        )
    if last_exc is not None:
        # Candidates existed but none parsed — could be prose-only output,
        # severely malformed JSON, or a single oversized envelope that hit
        # max_tokens before its first ``{`` even closed.
        raise ValueError(
            f"No parseable JSON in model output (last error at offset "
            f"{last_offset}: {last_exc.msg}; response length {src_len}). "
            f"Possible causes: max_tokens hit before envelope closed, "
            f"or model emitted prose without a JSON envelope."
        )
    raise ValueError("No JSON found in model output")


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
            raw_type = ent.get("entity_type") or ent.get("type", "concept")
            if isinstance(raw_type, list):
                raw_type = raw_type[0] if raw_type else "concept"
            fb_etype = fallback_entity_type()
            # entity_type is open (no Literal enforcement) — accept any
            # non-empty string so the model can emit rich types like
            # "product", "certification", "program", "paper", etc.
            # The schema YAML's entity_types list is a soft prior for
            # prompt examples; it does not gate the value here.
            type_str = str(raw_type).strip().lower() if raw_type else ""
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

            # Filter self-loops (e.g. "KIT studied at KIT")
            if subject.lower() == obj.lower():
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


def _correct_entity_names(graph: SessionGraph, transcript: str) -> SessionGraph:
    """Correct STT-garbled entity names using assistant responses.

    When the user says "Frankford" but the assistant responds with "Frankfurt",
    the assistant's spelling is more likely correct. Fuzzy-match extracted
    entity names against tokens in assistant responses (Levenshtein distance ≤ 2).
    """
    # Collect tokens from assistant responses
    assistant_tokens: set[str] = set()
    for line in transcript.split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("[assistant]") or line_lower.startswith("assistant:"):
            # Extract words ≥ 4 chars (skip common words)
            prefix_len = line.index("]") + 1 if "]" in line else line.index(":") + 1
            words = line[prefix_len:].split()
            for w in words:
                clean = w.strip(".,!?;:'\"()")
                if len(clean) >= 4:
                    assistant_tokens.add(clean)

    if not assistant_tokens:
        return graph

    # Check each entity and relation object against assistant tokens
    corrections: dict[str, str] = {}
    for entity in graph.entities:
        correction = _find_correction(entity.name, assistant_tokens)
        if correction:
            corrections[entity.name] = correction

    for relation in graph.relations:
        correction = _find_correction(relation.object, assistant_tokens)
        if correction:
            corrections[relation.object] = correction

    if not corrections:
        return graph

    # Apply corrections
    for entity in graph.entities:
        if entity.name in corrections:
            logger.info("STT correction: %s → %s", entity.name, corrections[entity.name])
            entity.name = corrections[entity.name]

    for relation in graph.relations:
        if relation.object in corrections:
            relation.object = corrections[relation.object]
        if relation.subject in corrections:
            relation.subject = corrections[relation.subject]

    return graph


def _find_correction(name: str, candidates: set[str]) -> str | None:
    """Find a candidate with Levenshtein distance ≤ 2 from name.

    Returns the candidate if found, None otherwise.
    Only corrects if the name is NOT already in the candidates (no self-match).
    """
    if name in candidates:
        return None

    for candidate in candidates:
        if abs(len(name) - len(candidate)) > 2:
            continue
        dist = _levenshtein(name.lower(), candidate.lower())
        if 0 < dist <= 2:
            return candidate
    return None


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s) < len(t):
        return _levenshtein(t, s)
    if len(t) == 0:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr_row = [i + 1]
        for j, tc in enumerate(t):
            cost = 0 if sc == tc else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _validate_with_ha_context(graph: SessionGraph, ha_context: dict) -> SessionGraph:
    """Validate and boost extracted location facts using HA home context.

    - Location matching HA's configured home → boost confidence to 1.0
    - Location matching a zone (home, work) → boost confidence to 0.9
    - Location with no HA connection → leave as-is (let LLM validator decide)

    This is a mechanical check — no LLM call.
    """
    location_name = ha_context.get("location_name", "").lower()
    zones = {z.lower() for z in ha_context.get("zones", [])}
    areas = {a.lower() for a in ha_context.get("areas", [])}
    all_known = zones | areas
    if location_name:
        all_known.add(location_name)

    if not all_known:
        return graph

    location_predicates = {
        "lives_in",
        "lives_near",
        "born_in",
        "located_in",
        "home_location",
    }

    for relation in graph.relations:
        if relation.predicate not in location_predicates:
            continue

        obj_lower = relation.object.lower()

        # Check if extracted location matches HA home
        if location_name and (location_name in obj_lower or obj_lower in location_name):
            logger.info(
                "HA validation: %s matches home location '%s' → confidence 1.0",
                relation.object,
                ha_context["location_name"],
            )
            relation.confidence = 1.0
            continue

        # Check zones and areas
        for known in all_known:
            if known in obj_lower or obj_lower in known:
                logger.info(
                    "HA validation: %s matches known location '%s' → confidence 0.9",
                    relation.object,
                    known,
                )
                relation.confidence = max(relation.confidence, 0.9)
                break

    return graph


_DEFAULT_ANONYMIZATION_PROMPT = """\
Anonymize the following extracted personal facts AND the conversation transcript \
by replacing all identifying information with category-prefixed placeholders.

Replace:
{replacement_rules}

Use the SAME placeholder for the SAME entity across facts and transcript. \
The mapping you return MUST contain every real name that appears in either input \
so the reverse mapping is total. If an entity appears in the transcript but not \
in any fact, still include it in the mapping (e.g. "my wife" mentioned but no \
fact yet — emit Person_2 in the mapping).

Keep predicates, relation_type, and confidence unchanged.

Facts to anonymize:
{facts_json}

Transcript to anonymize:
{transcript}

Return JSON in this exact format:
{{"anonymized_facts": [...], "anonymized_transcript": "...", \
"mapping": {{"original_name": "Person_1", ...}}}}

The mapping MUST use real names as keys and placeholders as values — \
the local de-anonymization step reverses this to recover real names.
"""

# Two-stage SOTA pipeline: enrichment first, then plausibility filtering.
# Each stage has a single responsibility and a separate prompt — combining
# them in one call (the previous "noise_filter" prompt) led to the LLM
# expanding scope at the same time as filtering, producing inflated counts
# and self-referential schema artifacts.

_DEFAULT_ENRICHMENT_PROMPT = """\
Review these extracted personal facts (anonymized — placeholders like Person_1).

1. Resolve coreference ("my wife" → married_to relation).
2. Split compound facts into individual relations.
3. Canonicalize symmetric predicates: emit only ONE direction. For predicates \
   like friend_of, married_to, sibling_of, colleague_of, neighbor_of, knows, \
   workout_partner_of, met_with — if both (A, p, B) and (B, p, A) are present, \
   drop the one where subject > object lexicographically. Asymmetric predicates \
   (parent_of/child_of, manages/reports_to) keep both directions if both stated.

You may ADD new facts that follow directly from the resolved coreference.
Do NOT drop facts for other reasons — a separate plausibility filter handles removal.

Conversation transcript (anonymized):
{transcript}

Extracted facts (anonymized):
{facts_json}

Return ONLY a JSON array of facts. Each fact: subject, predicate, object, \
relation_type, confidence. If none, return [].
"""

# NOTE: keep this template byte-equivalent to ``configs/prompts/sota_plausibility.txt``.
# ``tests/test_prompts_contract.py::test_inline_default_matches_file`` enforces parity.
# The file uses long markdown paragraphs; we preserve them via implicit string
# concatenation so Ruff E501 doesn't force line-wraps that alter the rendered text.
# fmt: off
_DEFAULT_PLAUSIBILITY_PROMPT = (
    "You are filtering enriched personal facts from a voice assistant conversation. "  # noqa: E501
    "Inputs may be anonymized (placeholders like Person_1) or real-named — apply the same rules either way.\n"  # noqa: E501
    "\n"
    "**Default: KEEP.** Silent data loss is worse than a noisy graph — the next session will correct noise, "  # noqa: E501
    "but a dropped real fact is gone forever. Only drop a fact when one of the rules below matches unambiguously. "  # noqa: E501
    "When uncertain, KEEP.\n"
    "\n"
    "Do NOT add new facts. Do NOT modify subject, predicate, object, relation_type, "  # noqa: E501
    "or confidence of facts you keep.\n"
    "\n"
    "## DROP rules (each is a lexical pattern — no judgment calls)\n"
    "\n"
    "**R1. Self-loop.** `subject` and `object` are the same string (case-insensitive), "  # noqa: E501
    "regardless of predicate.\n"
    "\n"
    "**R2. Name-swap pair.** Both `A has_name B` and `B has_name A` are present "  # noqa: E501
    "(also `named`, `is`, `equals`). Drop both.\n"
    "\n"
    "**R3. Role leak.** Subject or object is exactly one of: "
    '"Assistant", "User", "Speaker", "the bot", "the model". '
    'Note: "Person_1" / "City_1" / "Org_1" / "Thing_1" are valid entity placeholders, '  # noqa: E501
    "NOT role leaks.\n"
    "\n"
    "**R4. Unresolved placeholder in real-name input.** When the input is real-named "  # noqa: E501
    "(no `Person_N` / `City_N` / `Country_N` / `Org_N` / `Thing_N` placeholders expected in the transcript), "  # noqa: E501
    "drop any fact whose subject or object still matches "
    r"`^(Person|City|Country|Org|Thing)_\d+$`."  # noqa: E501
    "\n"
    "\n"
    "**R5. Empty / sentinel object.** Object is exactly one of: "
    '"", "Unknown", "None", "Various", "Something", "N/A".\n'
    "\n"
    "**R6. System entity ID.** Subject or object contains a dot-separated HA-style identifier "  # noqa: E501
    "(e.g. `media_player.sonos_office`, `sensor.temperature_kitchen`).\n"
    "\n"
    "## Examples\n"
    "\n"
    'Input fact: `{{"subject": "Person_1", "predicate": "has_name", "object": "Person_1"}}` → DROP (R1)\n'  # noqa: E501
    'Input fact: `{{"subject": "Alex", "predicate": "lives_in", "object": "Portland"}}` → KEEP\n'  # noqa: E501
    'Input fact: `{{"subject": "Assistant", "predicate": "responded_to", "object": "Alex"}}` → DROP (R3)\n'  # noqa: E501
    'Input fact: `{{"subject": "Alex", "predicate": "owns", "object": "Person_4"}}` → DROP (R4, real-name input)\n'  # noqa: E501
    'Input fact: `{{"subject": "Alex", "predicate": "controls", "object": "media_player.sonos_office"}}` → DROP (R6)\n'  # noqa: E501
    'Input fact: `{{"subject": "Alex", "predicate": "likes", "object": "Uptown Funk"}}` → KEEP\n'  # noqa: E501
    'Input fact: `{{"subject": "Alex", "predicate": "is_from", "object": "Unknown"}}` → DROP (R5)\n'  # noqa: E501
    "\n"
    "## Input\n"
    "\n"
    "Conversation transcript:\n"
    "{transcript}\n"
    "\n"
    "Enriched facts:\n"
    "{facts_json}\n"
    "\n"
    "Return ONLY a JSON array of surviving facts, schema unchanged. "
    "If all facts survive, return them all. If no facts survive, return [].\n"
)
# fmt: on


# Registry of SOTA plausibility validators that see only anonymized data.
# Keyed by the provider name callers pass as `plausibility_judge`.
# "auto" and "off" are NOT in this registry — they are handled by the
# deanon-stage (local judge) path; checking `judge in _PLAUSIBILITY_VALIDATORS`
# before dispatching to the cloud prevents the "auto" crash on
# `PROVIDER_KEY_ENV.get("auto")`.
#
# NOTE: This dict is duplicated from scripts/dev/compare_extraction.py::VALIDATORS.
# Both should stay in sync. TODO (PR2): move to a shared module to remove duplication.
_PLAUSIBILITY_VALIDATORS: dict[str, dict] = {
    "claude": {
        "type": "cloud",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "key_env": "ANTHROPIC_API_KEY",
    },
}


def _fallback_plausibility_on_raw(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    reason: str,
    *,
    speaker_name: str | None = None,
    speaker_id: str,
    role_aware_grounding: str = "off",
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
) -> SessionGraph:
    """Fallback pipeline path: run local plausibility + grounding on raw (unanonymized) facts.

    Used when anonymization fails entirely, when residual leaks after repair
    render the mapping non-canonical (no safe SOTA path), or when the full
    pipeline drops all relations.

    Steps (ported from scripts/dev/compare_extraction.py L722-795):
    1. Serialize graph.relations to fact dicts.
    2. Strip any residual placeholder tokens — records drops in diagnostics.
    3. Drop ungrounded facts (empty known_names — no mapping available).
    4. If non-empty, run local plausibility filter; keep raw on None return.
    5. Rebuild Relations, canonicalize symmetric predicates, filter entities.
    6. Record fallback_path in diagnostics.

    Args:
        speaker_id: Speaker store ID stamped onto every reconstructed
            ``Relation`` as provenance. Required — callers must always supply
            the session's speaker ID.

    Returns the modified graph in-place (graph.relations / graph.entities replaced).
    """
    from paramem.graph.schema import Relation

    raw_facts = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in graph.relations
    ]

    # Step 2: strip residual placeholders from raw facts (defensive)
    raw_facts, res_dropped = _strip_residual_placeholders(raw_facts)
    if res_dropped:
        graph.diagnostics["residual_dropped_facts"] = res_dropped
        logger.warning(
            "_fallback_plausibility_on_raw: dropped %d fact(s) with residual placeholders",
            len(res_dropped),
        )

    # Step 3: grounding gate with empty known_names (no mapping available)
    raw_facts, ungrounded, would_drop = _apply_grounding_gate(
        raw_facts,
        transcript,
        set(),
        speaker_name=speaker_name,
        mode=role_aware_grounding,
    )
    if ungrounded:
        graph.diagnostics["ungrounded_dropped_facts"] = ungrounded
        logger.warning(
            "_fallback_plausibility_on_raw: dropped %d ungrounded fact(s)",
            len(ungrounded),
        )
    if would_drop:
        graph.diagnostics["role_aware_would_drop"] = would_drop

    # Step 4: local plausibility filter (uses real names)
    if raw_facts and model is not None and tokenizer is not None:
        filtered = _local_plausibility_filter(
            raw_facts,
            transcript,
            model,
            tokenizer,
            max_tokens=plausibility_max_tokens,
            temperature=_DEFAULT_FILTER_TEMPERATURE,
        )
        if filtered is not None:
            pre = len(raw_facts)
            raw_facts = filtered
            dropped_count = pre - len(raw_facts)
            if dropped_count:
                graph.diagnostics["plausibility_dropped"] = dropped_count
                graph.diagnostics["plausibility_judge_actual"] = "local_fallback"

    # Step 5: rebuild Relations
    kept_relations = []
    for fact in raw_facts:
        try:
            kept_relations.append(
                Relation(
                    subject=fact.get("subject", ""),
                    predicate=fact.get("predicate", ""),
                    object=fact.get("object", ""),
                    relation_type=fact.get("relation_type", "factual"),
                    confidence=float(fact.get("confidence", 1.0)),
                    speaker_id=speaker_id,
                )
            )
        except Exception:
            continue

    kept_relations = _canonicalize_symmetric_predicates(kept_relations)
    kept_names = {r.subject for r in kept_relations} | {r.object for r in kept_relations}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    graph.relations = kept_relations

    # Step 6: record fallback path
    graph.diagnostics["fallback_path"] = reason
    logger.info(
        "_fallback_plausibility_on_raw: reason=%r, %d relation(s) surviving",
        reason,
        len(kept_relations),
    )
    return graph


def _sota_pipeline(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    speaker_id: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    endpoint: str | None = None,
    ner_check: bool = False,
    ner_model: str = "en_core_web_sm",
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    verify_anonymization: bool = True,
    speaker_name: str | None = None,
    role_aware_grounding: str = "off",
    pii_scope: set[str] | frozenset[str] | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
) -> SessionGraph:
    """Enrich extraction via local anonymization → SOTA enrichment → plausibility → de-anonymize.

    Stages:
    1. Local anonymize    → facts + transcript with placeholders (one total mapping)
    1d. Forward-path privacy guard (verify_anonymization=True): detect and repair leaks.
        Residual leak after repair: fact-level filter + skip SOTA, OR fallback to raw
        plausibility if mapping is non-canonical.
    2. SOTA enrichment    → coreference resolution + compound splitting + symmetric dedup
    3a. Plausibility on anonymized data (plausibility_stage="anon", SOTA judge)
    3b. De-anonymize + preserve pre-sweep snapshot
    3c. Residual placeholder sweep
    3d. Grounding gate
    3e. Plausibility on de-anonymized data (plausibility_stage="deanon", local judge)
    4. Build Relations + entity type rebuild + symmetric canonicalization
    5. All-dropped safety net → fallback to raw plausibility

    Falls back gracefully at every stage. Endpoint is forwarded for self-hosted
    OpenAI-compatible providers.

    Plausibility judges:
    - "auto"  → local model at deanon stage (zero cloud cost, privacy-safe)
    - "off"   → disable plausibility entirely
    - any SOTA provider name (e.g. "claude") → cloud judge at anon stage
    - "anthropic", "openai", "google", etc. → cloud judge at anon stage
      (must be combined with plausibility_stage="anon" to avoid PII exfiltration)
    """
    import os

    key_env_name = PROVIDER_KEY_ENV.get(provider)
    if key_env_name is None:
        logger.warning("Unsupported SOTA provider %r — skipping enrichment", provider)
        return graph
    api_key = os.environ.get(key_env_name, "")
    # Collect ALL config gaps before returning so a single warning surfaces
    # everything missing — avoids the "fix the key, then discover the endpoint
    # was also missing on the next run" loop.
    gaps = []
    if not api_key:
        gaps.append(f"{key_env_name} env var")
    if (
        provider in OPENAI_COMPAT_PROVIDERS
        and not endpoint
        and not OPENAI_COMPAT_ENDPOINTS.get(provider)
    ):
        gaps.append(f"endpoint for provider {provider!r}")
    if gaps:
        logger.info("Skipping SOTA enrichment — missing config: %s", ", ".join(gaps))
        return graph

    original_count = len(graph.relations)

    # Step 1: Anonymize facts AND transcript via local model in one call —
    # mapping is total over everything that will reach the SOTA stage.
    _vram_snapshot(f"sota_pipeline_entry session={graph.session_id}")
    anon_facts, mapping, anon_transcript = _anonymize_with_local_model(
        graph, model, tokenizer, transcript=transcript, max_tokens=max_tokens
    )
    _vram_snapshot(f"after_anonymize session={graph.session_id}")
    if anon_facts is None:
        logger.warning("Anonymization failed — falling back to raw plausibility")
        graph.diagnostics["anonymize"] = "failed"
        return _fallback_plausibility_on_raw(
            graph,
            transcript,
            model,
            tokenizer,
            "anon_failed",
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            role_aware_grounding=role_aware_grounding,
            max_tokens=max_tokens,
            plausibility_max_tokens=plausibility_max_tokens,
        )
    if not anon_facts:
        logger.info("Anonymization produced 0 facts — skipping SOTA pipeline")
        graph.diagnostics["grounding_gate"] = "no_input"
        graph.diagnostics["anonymize"] = "ok"
        graph.relations = []
        graph.entities = []
        return graph
    # Canonicalize mapping direction before any downstream use.
    mapping, norm_stats = _normalize_anonymization_mapping(mapping)
    if norm_stats["dropped"]:
        graph.diagnostics["mapping_ambiguous_dropped"] = norm_stats["dropped"]

    # Bug A fix: guarantee the speaker's display name is in the mapping so the
    # forward-path verifier does not flag bare first-name occurrences as leaks.
    # The extended mapping is propagated into anon_transcript via the
    # re-anonymization step below so every form of the name is replaced before
    # the verifier sees the output.
    if speaker_name:
        mapping = _ensure_speaker_name_in_mapping(mapping, speaker_name)

    # Bug B fix: recover forward-mapping entries the anonymizer emitted in
    # anonymized_facts but forgot to include in the returned mapping dict.
    # Without this, Product_1-style placeholders have no reverse entry and are
    # dropped by the residual sweep at de-anonymization time.
    mapping = _recover_missing_placeholder_mappings(mapping, anon_facts, graph.relations)

    # (Re-)build anonymized transcript AND facts from the ORIGINAL transcript
    # and relations using the now-complete mapping.  This covers three cases:
    # 1. LLM did not return an anonymized transcript (backward-compat fallback).
    # 2. Bug A extension added a new speaker-name entry — bare first-name tokens
    #    that the LLM left in the transcript or in fact subject/object fields
    #    must now be replaced before the verifier runs.
    # 3. Bug B extension added recovered Product_N-style entries — those entries
    #    already appear correctly in anon_facts (the LLM used Product_1 there);
    #    the transcript rebuild ensures any original name remaining in the
    #    transcript is also replaced.
    # Always running _anonymize_transcript is safe: it is idempotent when the
    # LLM already produced a clean transcript (placeholders are not word-chars
    # and therefore not matched as keys by _substitute_whole_words).
    anon_transcript = _anonymize_transcript(transcript, mapping)
    # Apply the (potentially extended) mapping to fact subject/object fields to
    # ensure no bare real-name tokens remain after Bug A seeding.
    anon_facts = [
        {
            **f,
            "subject": _substitute_whole_words(str(f.get("subject", "")), mapping),
            "object": _substitute_whole_words(str(f.get("object", "")), mapping),
        }
        if isinstance(f, dict)
        else f
        for f in anon_facts
    ]

    # Forward-path privacy guard: verify no real name leaked past anonymization
    # before sending anything to the cloud. On leak, attempt deterministic
    # repair (extend mapping for missed names, drop triples for hallucinated
    # ones). If residual leaks remain after repair:
    #   - mapping canonical → fact-level filter, skip SOTA, continue locally.
    #   - mapping non-canonical → fallback to raw plausibility (cannot safely repair).
    graph.diagnostics["anonymize"] = "ok"
    _skip_sota = False
    if verify_anonymization:
        extra_pii = (
            extract_pii_names_with_ner(transcript, ner_model, pii_scope=pii_scope)
            if ner_check
            else None
        )
        leaked = verify_anonymization_completeness(
            graph,
            mapping,
            anon_facts,
            anon_transcript,
            extra_pii_names=extra_pii,
            pii_scope=pii_scope,
        )
        if leaked:
            if _mapping_is_canonical(mapping):
                logger.info("Repairing %d leaked name(s): %s", len(leaked), leaked[:5])
                anon_facts, mapping, anon_transcript, repair_status = _repair_anonymization_leaks(
                    graph,
                    mapping,
                    anon_facts,
                    anon_transcript,
                    transcript,
                    leaked,
                    extra_pii_types=extra_pii,
                )
                logger.info(
                    "Repair: missed_fixed=%d hallucinated_dropped=%d",
                    repair_status["missed_fixed"],
                    repair_status["hallucinated_dropped"],
                )
                leaked = verify_anonymization_completeness(
                    graph,
                    mapping,
                    anon_facts,
                    anon_transcript,
                    extra_pii_names=extra_pii,
                    pii_scope=pii_scope,
                )
                if leaked:
                    # Residual leak after repair with canonical mapping:
                    # drop facts that reference leaked names, skip SOTA, continue locally.
                    leaked_lc = {n.lower() for n in leaked}
                    pre_filter = len(anon_facts)
                    anon_facts = [
                        f
                        for f in anon_facts
                        if not (
                            str(f.get("subject", "")).lower() in leaked_lc
                            or str(f.get("object", "")).lower() in leaked_lc
                        )
                    ]
                    dropped_count = pre_filter - len(anon_facts)
                    graph.diagnostics["residual_leaked_triples_dropped"] = dropped_count
                    graph.diagnostics["residual_leaked"] = leaked[:10]
                    graph.diagnostics["anonymize"] = "leaked_repaired"
                    _skip_sota = True
                    logger.warning(
                        "Residual leaks after repair (%s); dropped %d triple(s) referencing "
                        "leaked names, skipping SOTA.",
                        leaked[:5],
                        dropped_count,
                    )
            else:
                # Non-canonical mapping — cannot safely repair. Fall back to raw plausibility.
                logger.warning(
                    "Residual leaks with non-canonical mapping (%s); falling back to raw "
                    "plausibility.",
                    leaked[:5],
                )
                graph.diagnostics["anonymize"] = "leaked_noncanonical"
                return _fallback_plausibility_on_raw(
                    graph,
                    transcript,
                    model,
                    tokenizer,
                    "anon_leaked_noncanonical",
                    speaker_name=speaker_name,
                    speaker_id=speaker_id,
                    role_aware_grounding=role_aware_grounding,
                    max_tokens=max_tokens,
                    plausibility_max_tokens=plausibility_max_tokens,
                )

    # Step 2: SOTA enrichment — coreference + compound splitting + safe reification.
    # Skipped when _skip_sota=True (residual leak after repair with canonical mapping).
    updated_anon_transcript = None
    _sota_raw = None
    sota_bindings: dict[str, str] = {}
    if _skip_sota:
        # Skip SOTA — use filtered anon_facts as-is. No new bindings since
        # SOTA didn't run.
        enriched_anon = anon_facts
        logger.info(
            "Skipping SOTA enrichment (residual leak path); using %d fact(s)", len(anon_facts)
        )
    else:
        # Send anon facts and transcript to SOTA as the local anonymizer
        # produced them. The SOTA prompt's convention is "anonymizer
        # placeholders are bare; only new entities introduced by SOTA use
        # braced form (`{Prefix_N}`)". SOTA also returns explicit bindings
        # for any braced placeholders it minted, so de-anonymization is
        # pure dict substitution downstream — no transcript diff, no LLM
        # call, no regex post-processing.
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
        )
        if _sota_info:
            graph.diagnostics["sota_call_info"] = _sota_info
        if enriched_anon is None:
            logger.warning("SOTA enrichment failed — keeping pre-enrichment facts")
            enriched_anon = anon_facts

        if not enriched_anon:
            logger.info("SOTA enrichment removed all relations")

    # Step 3a: Plausibility on anonymized data (SOTA judge, stage="anon").
    # Only runs when: explicit SOTA provider, plausibility_stage=="anon", not _skip_sota,
    # and enriched_anon is non-empty.
    # Guard: use `plausibility_judge in _PLAUSIBILITY_VALIDATORS` (NOT != "off") —
    # "auto" is not in the registry and would crash PROVIDER_KEY_ENV.get("auto").
    if (
        plausibility_stage == "anon"
        and plausibility_judge in _PLAUSIBILITY_VALIDATORS
        and not _skip_sota
        and enriched_anon
    ):
        pv_info = _PLAUSIBILITY_VALIDATORS[plausibility_judge]
        pv_key = os.environ.get(pv_info["key_env"], "")
        if pv_key:
            plaus_facts, plaus_raw = _plausibility_filter_with_sota(
                enriched_anon,
                pv_key,
                provider=pv_info["provider"],
                filter_model=pv_info["model_id"],
                anon_transcript=anon_transcript,
                endpoint=pv_info.get("endpoint"),
                max_tokens=max_tokens,
                temperature=_DEFAULT_FILTER_TEMPERATURE,
            )
            # Cloud round-trip can take 30–90s during which the WSL2 GPU
            # goes idle and the next local CUDA op fails with
            # "device not ready". Wake + settle before the deanon-stage
            # local plausibility filter that follows below.
            _wait_for_gpu_ready()
            if plaus_facts is not None:
                pre_plaus = len(enriched_anon)
                enriched_anon = plaus_facts
                dropped_plaus = pre_plaus - len(enriched_anon)
                graph.diagnostics["plausibility"] = "anon"
                graph.diagnostics["plausibility_dropped"] = dropped_plaus
                graph.diagnostics["plausibility_judge_actual"] = plausibility_judge
                if plaus_raw:
                    graph.diagnostics["sota_plausibility_raw_response"] = plaus_raw
                logger.info(
                    "Anon-stage plausibility (%s): %d → %d facts (%d dropped)",
                    plausibility_judge,
                    pre_plaus,
                    len(enriched_anon),
                    dropped_plaus,
                )
            else:
                logger.warning("Anon-stage plausibility call failed — keeping enriched facts")
        else:
            logger.warning(
                "Anon-stage plausibility: no API key for %r — skipping", plausibility_judge
            )

    # Empty-check guard (compare L1019-1028): if enriched_anon is empty after
    # anon-stage plausibility (or was already empty), return early.
    if not enriched_anon:
        logger.info("No facts remain after anon-stage plausibility — returning empty graph")
        graph.diagnostics["grounding_gate"] = "no_input"
        graph.relations = []
        graph.entities = []
        return graph

    # Step 3b: De-anonymize via state-machine substitution.
    #
    # Uses the anonymizer mapping (real_name -> placeholder) plus SOTA's
    # explicit ``new_entity_bindings`` (placeholder -> real_text) for any
    # entities SOTA introduced. Pure dict substitution — no LLM call,
    # no transcript diff, no regex. Replaces the previous regex chain
    # (``_brace_placeholders_in_text/_in_facts`` + ``_extract_sota_bindings``
    # + ``_strip_placeholder_braces`` + reverse-mapping substitution).
    from paramem.graph.schema import Entity, Relation

    # Reverse anonymizer mapping (placeholder -> real_name). Used by the
    # entity-rebuild loop later to derive entity_type from the placeholder
    # prefix.
    reverse_mapping = {
        v: k for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str) and k
    }

    deanon_input_count = len(enriched_anon)
    deanon_facts, dropped_facts = _apply_bindings(enriched_anon, mapping, sota_bindings)
    if dropped_facts:
        graph.diagnostics["residual_dropped_facts"] = dropped_facts
        logger.warning(
            "Dropped %d fact(s) with residual placeholders post-substitution "
            "(missing SOTA binding or anonymizer leak).",
            len(dropped_facts),
        )
    deanon_dropped = deanon_input_count - len(deanon_facts)
    if deanon_dropped:
        logger.info(
            "De-anon: %d → %d facts (%d dropped)",
            deanon_input_count,
            len(deanon_facts),
            deanon_dropped,
        )

    # Step 3c+: Route scalar-valued objects (URLs, emails, phone numbers,
    # DOIs, version-tagged tool names like "ROS2") off the relation surface
    # and onto Entity.attributes of the subject.  Scalars are verbatim
    # identifiers, not paraphrasable concept phrases — they fail the
    # grounding gate's whole-word alpha-token rule because their alpha
    # portion is concatenated with digits in the transcript
    # ("username1234", "ROS2").  Routing them to attributes mirrors the
    # email/phone/linkedin path the local extractor already populates and
    # which the QA generator's _flatten_entity_attributes mints into keyed
    # pairs without grounding.  The projection is applied after the entity
    # rebuild step below so the subject entity survives pruning.
    scalar_facts, deanon_facts = _partition_scalar_facts(deanon_facts)
    if scalar_facts:
        graph.diagnostics["scalar_facts_projected"] = len(scalar_facts)

    # Step 3d: Grounding gate — every surviving fact's subject and object must be
    # grounded in the original transcript OR be a known real-name from the mapping.
    # Catches SOTA world-knowledge inferences (e.g. inferring "CIA" from a
    # transcript that only mentions "Langley") — those entities are dropped because
    # they never existed in the user's actual words.
    known_real_names = set(mapping.keys())
    deanon_facts, ungrounded, would_drop = _apply_grounding_gate(
        deanon_facts,
        transcript,
        known_real_names,
        speaker_name=speaker_name,
        mode=role_aware_grounding,
    )
    if ungrounded:
        graph.diagnostics["ungrounded_dropped_facts"] = ungrounded
        logger.warning(
            "Dropped %d fact(s) with entities ungrounded in the transcript "
            "(likely SOTA world-knowledge inference).",
            len(ungrounded),
        )
    if would_drop:
        graph.diagnostics["role_aware_would_drop"] = would_drop

    if _sota_raw:
        graph.diagnostics["sota_raw_response"] = _sota_raw
    if updated_anon_transcript:
        graph.diagnostics["sota_updated_transcript"] = updated_anon_transcript

    # Step 3e: Plausibility on de-anonymized data (local judge, stage="deanon").
    # Runs when plausibility_judge != "off" AND plausibility_stage == "deanon"
    # AND model/tokenizer are available (guard against tests that pass None).
    # "auto" resolves to the local model. Receives the ORIGINAL real-name transcript
    # (NOT anon_transcript) — privacy-critical when _skip_sota=True (leaked names
    # may still be in anon_transcript but are safe in the real transcript).
    if (
        plausibility_stage == "deanon"
        and plausibility_judge != "off"
        and deanon_facts
        and model is not None
        and tokenizer is not None
    ):
        _vram_snapshot(f"before_plausibility_deanon session={graph.session_id}")
        filtered_deanon = _local_plausibility_filter(
            deanon_facts,
            transcript,  # original real-name transcript — intentional, see docstring
            model,
            tokenizer,
            max_tokens=plausibility_max_tokens,
            temperature=_DEFAULT_FILTER_TEMPERATURE,
        )
        if filtered_deanon is not None:
            pre_deanon = len(deanon_facts)
            deanon_facts = filtered_deanon
            dropped_deanon = pre_deanon - len(deanon_facts)
            graph.diagnostics["plausibility"] = "deanon"
            graph.diagnostics["plausibility_dropped"] = (
                graph.diagnostics.get("plausibility_dropped", 0) + dropped_deanon
            )
            graph.diagnostics["plausibility_judge_actual"] = (
                plausibility_judge if plausibility_judge != "auto" else "local"
            )
            logger.info(
                "Deanon-stage plausibility (local): %d → %d facts (%d dropped)",
                pre_deanon,
                len(deanon_facts),
                dropped_deanon,
            )
        else:
            logger.warning("Deanon-stage plausibility call failed — keeping deanon facts")

    kept_relations = []
    validation_dropped: list[dict] = []
    for fact in deanon_facts:
        try:
            kept_relations.append(
                Relation(
                    subject=fact.get("subject", ""),
                    predicate=fact.get("predicate", ""),
                    object=fact.get("object", ""),
                    relation_type=fact.get("relation_type", "factual"),
                    confidence=float(fact.get("confidence", 1.0)),
                    speaker_id=speaker_id,
                )
            )
        except Exception as exc:
            validation_dropped.append(
                {
                    "subject": fact.get("subject", ""),
                    "predicate": fact.get("predicate", ""),
                    "object": fact.get("object", ""),
                    "relation_type": fact.get("relation_type", ""),
                    "reason": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            continue
    if validation_dropped:
        graph.diagnostics["pydantic_validation_dropped"] = validation_dropped
        logger.warning(
            "Dropped %d fact(s) at Relation schema validation "
            "(commonly: relation_type outside Literal set)",
            len(validation_dropped),
        )

    # Step 4: Deterministic safety net for symmetric-predicate canonicalization.
    # The enrichment prompt asks the LLM to drop the inverse direction; this
    # guards against the LLM leaving both. Local-only, no extra API call.
    kept_relations = _canonicalize_symmetric_predicates(kept_relations)

    # Step 5: All-dropped safety net — if every relation was dropped and the
    # original extraction had facts, fall back to raw plausibility so the
    # session does not yield zero facts due to anonymizer inconsistency.
    if not kept_relations and original_count > 0:
        logger.warning(
            "All %d relation(s) dropped by pipeline — triggering all_dropped fallback",
            original_count,
        )
        return _fallback_plausibility_on_raw(
            graph,
            transcript,
            model,
            tokenizer,
            "all_dropped",
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            role_aware_grounding=role_aware_grounding,
            max_tokens=max_tokens,
            plausibility_max_tokens=plausibility_max_tokens,
        )

    # Rebuild entity list from surviving + new relations.
    # Every relation endpoint must have a corresponding Entity record.
    # Entity type inference: known prefixes (Person, Org, City, ...) use the
    # configured anonymizer_prefix_to_type() mapping. Novel prefixes that SOTA
    # introduces (Project_1, Program_1, Paper_1, Certification_1, ...) derive
    # the type from the prefix itself — the prefix name IS the type name in
    # SOTA's brace-binding protocol. entity_type is open (no Literal), so the
    # derived type passes through.
    kept_names = (
        {r.subject for r in kept_relations}
        | {r.object for r in kept_relations}
        | {str(f.get("subject", "")) for f in scalar_facts if str(f.get("subject", "")).strip()}
    )
    existing_names = {e.name for e in graph.entities}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    closed_prefix_to_type = anonymizer_prefix_to_type()
    for name in kept_names - existing_names:
        entity_type = "concept"
        placeholder = reverse_mapping.get(name)
        if placeholder:
            prefix = placeholder.split("_")[0].lower()
            # Closed prefix → configured type. Novel prefix → use the prefix
            # itself as the type. Empty prefix or no placeholder → "concept".
            entity_type = closed_prefix_to_type.get(prefix) or prefix or "concept"
        graph.entities.append(Entity(name=name, entity_type=entity_type))

    # Project scalar-object facts onto subject Entity.attributes (see
    # _partition_scalar_facts call above for rationale).
    _project_scalar_facts_to_attributes(graph, scalar_facts)

    graph.relations = kept_relations

    added = len(kept_relations) - original_count
    logger.info(
        "SOTA enrichment: %d → %d relations (%+d)",
        original_count,
        len(kept_relations),
        added,
    )
    return graph


def _next_placeholder_index(mapping: dict, prefix: str) -> int:
    """Return the next free placeholder index for ``prefix`` within ``mapping``.

    Scans the mapping VALUES for existing ``Prefix_N`` tokens and returns
    ``max(N) + 1``.  Used by the speaker-name seeding and gap-recovery
    helpers to allocate non-colliding placeholder indices.

    Args:
        mapping: Current ``{real_name: placeholder}`` mapping (canonical form).
        prefix: Capitalised placeholder prefix, e.g. ``"Person"``.

    Returns:
        Integer index (≥ 1) that is not yet used by any mapping value.
    """
    max_n = 0
    for v in mapping.values():
        if not isinstance(v, str):
            continue
        if v.startswith(f"{prefix}_"):
            tail = v.split("_")[-1]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return max_n + 1


def _ensure_speaker_name_in_mapping(
    mapping: dict,
    speaker_name: str | None,
) -> dict:
    """Guarantee that the speaker's display name is covered by the anonymizer mapping.

    Bug A fix: the local anonymizer maps the full-name form (e.g.
    ``Alex Rivera → Person_1``) but leaves the bare first name (``Alex``)
    out of the mapping.  After :func:`_stamp_speaker_entity` renames the graph
    entity to the display name (``Alex``), the forward-path privacy verifier
    finds ``Alex`` in the transcript and flags it as a leak — triggering the
    residual-leak path, skipping SOTA enrichment, and ultimately dropping two
    facts that contain ``Product_1`` (which can only be resolved via the full
    SOTA enrichment path).

    This function is called once per pipeline run, AFTER
    :func:`_normalize_anonymization_mapping`, BEFORE the first
    :func:`verify_anonymization_completeness` call.  It extends the mapping
    in-place-safe fashion (returns a new dict) so the verifier and all
    downstream steps see a complete mapping that covers every form the speaker's
    name can take.

    Resolution strategy (in priority order):
    1. ``speaker_name`` already a key → no-op (mapping is complete).
    2. A key in ``mapping`` starts with ``speaker_name + " "`` (full-name
       variant, e.g. ``"Alex Rivera"``) → reuse its placeholder so both
       the bare and full forms map to the *same* ``Person_N`` token.
       De-anonymization then consistently produces the full name at every site.
    3. No existing key covers ``speaker_name`` → allocate a fresh ``Person_N``
       placeholder (type-safe: speakers are always persons).

    Args:
        mapping: Canonical ``{real_name: placeholder}`` mapping as returned
            by :func:`_normalize_anonymization_mapping`.
        speaker_name: Display name of the speaker (e.g. ``"Alex"`` or
            ``"Sam"``) as passed to :func:`_sota_pipeline`.  When ``None``
            or empty, the mapping is returned unchanged.

    Returns:
        Potentially-extended mapping (new dict; input is not mutated).
    """
    if not speaker_name:
        return mapping
    if speaker_name in mapping:
        return mapping

    new_mapping = dict(mapping)
    speaker_lower = speaker_name.lower()

    # Strategy 2: reuse the placeholder of any full-name key that starts with
    # the speaker's first name (e.g. "Alex Rivera" when speaker is "Alex").
    for key, placeholder in mapping.items():
        if not isinstance(key, str):
            continue
        if key.lower().startswith(speaker_lower + " "):
            new_mapping[speaker_name] = placeholder
            logger.debug(
                "Speaker-name seeding: %r reuses placeholder %r from full-name key %r",
                speaker_name,
                placeholder,
                key,
            )
            return new_mapping

    # Strategy 3: allocate a fresh Person_N placeholder.
    person_prefix = anonymizer_type_to_prefix().get("person", "Person")
    fresh_placeholder = f"{person_prefix}_{_next_placeholder_index(new_mapping, person_prefix)}"
    new_mapping[speaker_name] = fresh_placeholder
    logger.debug(
        "Speaker-name seeding: %r allocated fresh placeholder %r",
        speaker_name,
        fresh_placeholder,
    )
    return new_mapping


def _recover_missing_placeholder_mappings(
    mapping: dict,
    anon_facts: list[dict],
    original_relations: list,
) -> dict:
    """Recover anonymizer mapping entries that the model emitted in facts but omitted from mapping.

    Bug B fix: the local anonymizer may produce placeholder tokens in
    ``anonymized_facts`` (e.g. ``Product_1``) without including the corresponding
    forward mapping entry (e.g. ``Honda → Product_1``).  When this happens the
    reverse mapping used at de-anonymization time has no entry for ``Product_1``,
    so the placeholder survives the residual sweep and the fact is dropped.

    This function is called once per pipeline run, after
    :func:`_anonymize_with_local_model` and after
    :func:`_normalize_anonymization_mapping`.  It scans each ``anon_fact``
    field (subject, object) for placeholder-shaped tokens that are absent from
    ``mapping.values()`` and tries to recover the original value by comparing
    against the corresponding ``original_relations`` entry.

    Matching strategy:
    - Primary: positional (anon_facts[i] ↔ original_relations[i]).  The
      anonymizer is expected to preserve fact order.
    - The comparison is field-by-field: when anon_fact's ``subject`` is a
      bare placeholder not in ``mapping.values()`` and original_relation's
      ``subject`` is not a placeholder, add ``original → placeholder`` to the
      recovered mapping.

    Only bare placeholder tokens (matching :data:`_PLACEHOLDER_TOKEN_RE`)
    that are NOT already covered by ``mapping.values()`` are recovered.
    Braced SOTA-style tokens (``{Prefix_N}``) are intentionally skipped —
    those belong to the SOTA enrichment path and are handled separately.

    Args:
        mapping: Canonical ``{real_name: placeholder}`` mapping as returned
            by :func:`_normalize_anonymization_mapping` (and extended by
            :func:`_ensure_speaker_name_in_mapping`).
        anon_facts: Anonymized fact dicts returned by
            :func:`_anonymize_with_local_model`.
        original_relations: The ``graph.relations`` list BEFORE anonymization
            (list of :class:`~paramem.graph.schema.Relation` objects).

    Returns:
        Potentially-extended mapping (new dict; input is not mutated).
    """
    if not anon_facts or not original_relations:
        return mapping

    # Set of placeholder token strings already covered — no need to recover.
    covered_placeholders: set[str] = {
        str(v).lower() for v in mapping.values() if isinstance(v, str) and v
    }

    # Bare placeholder pattern (no braces, no anchoring) — finds embedded
    # tokens like "Product_1" inside "software for Product_1's Legend system".
    _bare_ph_re = re.compile(r"\b([A-Z][A-Za-z]*_\d+)\b")

    recovered: dict[str, str] = {}
    for i, anon_fact in enumerate(anon_facts):
        if not isinstance(anon_fact, dict):
            continue
        if i >= len(original_relations):
            break
        orig = original_relations[i]
        orig_subj = getattr(orig, "subject", "") or ""
        orig_obj = getattr(orig, "object", "") or ""
        anon_subj = str(anon_fact.get("subject", ""))
        anon_obj = str(anon_fact.get("object", ""))

        for anon_val, orig_val in ((anon_subj, orig_subj), (anon_obj, orig_obj)):
            if not anon_val or not orig_val:
                continue
            # Find all bare placeholders in the anon field.
            ph_tokens = _bare_ph_re.findall(anon_val)
            if not ph_tokens:
                continue
            for ph_token in ph_tokens:
                if ph_token.lower() in covered_placeholders:
                    continue
                # The original field must not already contain this placeholder.
                if ph_token in orig_val:
                    continue
                # Two recovery strategies:
                #
                # Case A — whole-field placeholder: the entire anon_val IS the
                # placeholder (e.g. anon_val="Org_1", orig_val="Acme Corp").
                # The original field is recovered directly.
                if anon_val.strip() == ph_token and orig_val not in mapping:
                    recovered[orig_val] = ph_token
                    covered_placeholders.add(ph_token.lower())
                    logger.debug(
                        "Gap recovery (whole-field): %r → %r (position %d)",
                        orig_val,
                        ph_token,
                        i,
                    )
                    break
                #
                # Case B — embedded placeholder: ph_token appears inside a
                # compound anon_val (e.g. "software for Product_1's Legend").
                # Recover the original word by finding the word in orig_val that
                # is absent from anon_val, is not itself a placeholder, and when
                # substituted into anon_val in place of ph_token reconstructs
                # orig_val exactly.
                orig_words = re.findall(r"\b[\w]+\b", orig_val)
                anon_words = set(re.findall(r"\b[\w]+\b", anon_val))
                candidates = [
                    w
                    for w in orig_words
                    if w not in anon_words
                    and not re.match(r"^[A-Z][A-Za-z]*_\d+$", w)
                    and w not in mapping
                ]
                for candidate in candidates:
                    # Verify: replacing ph_token with candidate in anon_val
                    # produces the original field (whole-word replacement).
                    reconstructed = _substitute_whole_words(anon_val, {ph_token: candidate})
                    if reconstructed == orig_val:
                        recovered[candidate] = ph_token
                        covered_placeholders.add(ph_token.lower())
                        logger.debug(
                            "Gap recovery (embedded): %r → %r (position %d, verified)",
                            candidate,
                            ph_token,
                            i,
                        )
                        break

    if not recovered:
        return mapping

    logger.info(
        "Mapping gap recovery: added %d missing entr%s: %s",
        len(recovered),
        "y" if len(recovered) == 1 else "ies",
        list(recovered.items())[:5],
    )
    new_mapping = dict(mapping)
    new_mapping.update(recovered)
    return new_mapping


def _repair_anonymization_leaks(
    graph: SessionGraph,
    mapping: dict,
    anon_facts: list[dict],
    anon_transcript: str,
    original_transcript: str,
    leaked: list[str],
    extra_pii_types: dict[str, str] | None = None,
) -> tuple[list[dict], dict, str, dict]:
    """Deterministic repair of anonymization leaks — no LLM call.

    For each leaked name:
    - If the name appears in the original transcript (whole-word, case-insensitive),
      classify as "missed": extend mapping with the next free placeholder of the
      right PII type (person→Person_N, place→City_N), rewrite anon_facts and
      anon_transcript via the extended mapping.
    - Otherwise classify as "hallucinated": drop every triple in anon_facts
      whose subject or object matches the leaked name. Mapping is not extended.

    ``extra_pii_types`` is a ``{name: "person"|"place"}`` mapping
    contributed by NER (see :func:`extract_pii_names_with_ner`).
    Consulted only when the extractor's own ``type_by_name`` has no
    entry for a leaked name — without this fallback the type defaults
    to ``"person"`` regardless of NER's classification, producing
    misclassified placeholders (e.g. ``Berlin → Person_4`` instead of
    ``Berlin → City_1``).  De-anonymization on the return path then
    fails to swap the placeholder back to the real city name because
    the mapping direction is wrong by category.  Extractor types win
    on collision; NER is the fallback, not the override.

    Precondition: mapping must be in canonical {real: placeholder} direction.
    Caller checks `_mapping_is_canonical(mapping)` and skips repair otherwise.

    Returns: (repaired_facts, extended_mapping, repaired_transcript, repair_status)
    where repair_status = {"missed_fixed", "hallucinated_dropped", "residual_dropped"}.
    """
    type_by_name = {e.name: e.entity_type for e in graph.entities}
    status = {"missed_fixed": 0, "hallucinated_dropped": 0, "residual_dropped": 0}

    new_mapping = dict(mapping)
    facts = [dict(f) for f in anon_facts if isinstance(f, dict)]

    # Compute next-index allocator per prefix from current mapping values.
    def _next_index(prefix: str) -> int:
        max_n = 0
        for v in new_mapping.values():
            if not isinstance(v, str):
                continue
            if v.startswith(f"{prefix}_"):
                tail = v.split("_")[-1]
                if tail.isdigit():
                    max_n = max(max_n, int(tail))
        return max_n + 1

    hallucinated: set[str] = set()
    for name in leaked:
        if not name:
            continue
        in_transcript = _contains_whole_word(original_transcript or "", name, case_insensitive=True)
        if not in_transcript:
            hallucinated.add(name)
            continue
        # Missed — allocate placeholder based on declared type.  Extractor
        # type wins; fall back to NER (extra_pii_types) when extractor has
        # no opinion; final default to "person" preserves prior behaviour.
        ner_type = (extra_pii_types or {}).get(name)
        etype = (type_by_name.get(name) or ner_type or "person").lower()
        prefix = anonymizer_type_to_prefix().get(etype)
        if prefix is None:
            # Missed — allocate placeholder based on declared type. Types with a
            # primary_for_type prefix in configs/schema.yaml (person→Person, place→
            # City, organization→Org, concept→Thing) get a fresh placeholder of
            # that type. Types without a primary prefix (event, preference) are
            # treated as hallucinated — there is no PII-scope anchor to bind to.
            hallucinated.add(name)
            continue
        placeholder = f"{prefix}_{_next_index(prefix)}"
        new_mapping[name] = placeholder
        status["missed_fixed"] += 1

    # Drop hallucinated-referencing triples from anon_facts.
    if hallucinated:
        hallu_lc = {h.lower() for h in hallucinated}
        kept = []
        for f in facts:
            s = str(f.get("subject", ""))
            o = str(f.get("object", ""))
            if s.lower() in hallu_lc or o.lower() in hallu_lc:
                status["hallucinated_dropped"] += 1
                continue
            kept.append(f)
        facts = kept

    # Field-level rewrite of subject/object for missed names, then mechanical
    # transcript re-anonymization with the extended mapping.
    missed_names = {n for n in leaked if n not in hallucinated}
    if missed_names:
        # Build a focused mapping for just the missed names; reuse the
        # shared _substitute_whole_words helper (longest-first internally,
        # word-boundary anchored — same primitive _anonymize_transcript uses).
        missed_mapping = {name: new_mapping[name] for name in missed_names}
        for f in facts:
            s = f.get("subject", "")
            o = f.get("object", "")
            if isinstance(s, str):
                f["subject"] = _substitute_whole_words(s, missed_mapping)
            if isinstance(o, str):
                f["object"] = _substitute_whole_words(o, missed_mapping)
        anon_transcript = _anonymize_transcript(original_transcript, new_mapping)

    return facts, new_mapping, anon_transcript, status


_PLACEHOLDER_TOKEN_RE = re.compile(r"\{(\w+_\d+)\}|\b([A-Z][A-Za-z]*_\d+)\b")


def _normalize_for_grounding(text: str) -> str:
    """Lowercase and underscore-to-space normalise for transcript comparisons."""
    return (text or "").replace("_", " ").lower()


def _extract_user_spans(transcript: str, *, speaker_name: str | None = None) -> str:
    """Concatenate every line of user-authored content from ``transcript``.

    Walks lines as a state machine: a role-prefix line (``[user]`` /
    ``user:`` / ``[assistant]`` / ``assistant:`` / ``{speaker_name}:``)
    sets the current role, and **subsequent unmarked lines inherit it
    until the next role transition**.  Pre-marker content is dropped
    (state starts as ``None``), preserving the no-context-no-trust
    invariant: when no role can be established for a line, it is not
    treated as user content.

    For a single-turn document chunk, ``SessionBuffer._format_turns``
    prepends a one-time ``{speaker_name}: `` to the first line and the
    rest of the document follows with its natural line breaks.  The
    state machine catches the first line as a role transition into
    ``user`` and inherits that role across the body.  This also fixes
    multi-line user turns in real dialogue (``user: line one\\nline
    two``) which the previous per-line predicate silently truncated.

    The asymmetric mirror is :func:`_correct_entity_names` (assistant
    blocks).  ``speaker_name`` of ``"assistant"`` is ignored — that
    label is reserved for the model marker and never accepted as a user
    speaker.
    """
    speaker_prefix: str | None = None
    speaker_prefix_skip = 0
    if speaker_name and speaker_name.lower() != "assistant":
        speaker_prefix = speaker_name.lower() + ":"
        # Skip by len(speaker_name)+1 (the colon) on the original-case line.
        # Lowercasing can change byte length for some Unicode (Turkish İ, German
        # ẞ); using original-case length keeps the slice index correct.
        speaker_prefix_skip = len(speaker_name) + 1

    out: list[str] = []
    current_role: str | None = None  # no role established → no capture
    for line in (transcript or "").split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        text_after: str
        if lower.startswith("[user]"):
            current_role = "user"
            prefix_len = stripped.index("]") + 1
            text_after = stripped[prefix_len:].lstrip()
        elif lower.startswith("[assistant]"):
            current_role = "assistant"
            prefix_len = stripped.index("]") + 1
            text_after = stripped[prefix_len:].lstrip()
        elif lower.startswith("user:"):
            current_role = "user"
            text_after = stripped[len("user:") :].lstrip()
        elif lower.startswith("assistant:"):
            current_role = "assistant"
            text_after = stripped[len("assistant:") :].lstrip()
        elif speaker_prefix and lower.startswith(speaker_prefix):
            current_role = "user"
            text_after = stripped[speaker_prefix_skip:].lstrip()
        else:
            # Unmarked line — inherit the current role.
            text_after = stripped
        if current_role == "user" and text_after:
            out.append(text_after)
    return "\n".join(out)


def _is_scalar_value(value: str) -> bool:
    """True iff ``value`` is a verbatim identifier rather than a content phrase.

    Scalars belong on ``Entity.attributes`` (alongside email/phone/linkedin
    extracted by the local extractor), where the QA generator's
    :func:`_flatten_entity_attributes` mints keyed pairs without grounding.
    Routing them through the grounding gate as relations is wrong: the
    gate's whole-word alpha-token rule rejects values whose alpha portion
    lives concatenated with digits in the source transcript
    (``username1234``, ``ROS2``, ``H100``), even though the scalar is
    verbatim in the text.

    Detection is structural — no regex.  Recognised shapes:

    * **Phone**: ≥7 digits, characters drawn only from
      ``digits + " +()-."``.  Spaces are permitted only inside this class
      (phone numbers are the one multi-token scalar).
    * **Email**: contains ``@`` and a ``.`` after the ``@``, no whitespace.
    * **URL with scheme**: starts with ``http://`` / ``https://`` /
      ``ftp://``.
    * **URL without scheme**: contains ``/`` and the part before the
      first ``/`` looks like a domain (alphanumeric + dots/dashes,
      contains at least one ``.``).
    * **DOI**: starts with ``10.`` and contains ``/``.
    * **Version-tagged identifier**: contains both letters and digits,
      no spaces, characters drawn only from ``alnum + "/-_+."``
      (e.g. ``ROS2``, ``H100``, ``IPv6``, ``ROS2/Gazebo``, ``x86_64``).
    """
    s = (value or "").strip()
    if not s:
        return False
    digit_count = sum(c.isdigit() for c in s)
    # Phone: ≥7 digits, only phone-shaped chars (the only multi-token scalar).
    if digit_count >= 7 and all(c.isdigit() or c in "+()-. " for c in s):
        return True
    # After the phone exemption, multi-word means content phrase.
    if any(c.isspace() for c in s):
        return False
    # Email: "@" with a domain ending in a TLD-like dot suffix.
    if "@" in s:
        domain = s.rsplit("@", 1)[-1]
        if "." in domain and len(domain) > 2:
            return True
    lower = s.lower()
    # URL with scheme.
    if lower.startswith(("http://", "https://", "ftp://")):
        return True
    # URL without scheme: domain.tld[/path].
    if "/" in s:
        head = s.split("/", 1)[0]
        if "." in head and head.replace(".", "").replace("-", "").isalnum():
            return True
    # DOI.
    if s.startswith("10.") and "/" in s and digit_count > 0:
        return True
    # Version-tagged identifier.
    if digit_count > 0 and any(c.isalpha() for c in s):
        if all(c.isalnum() or c in "/-_+." for c in s):
            return True
    return False


def _partition_scalar_facts(facts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split facts by object-shape into ``(scalar, non_scalar)``.

    Scalars are projected onto ``Entity.attributes`` of the subject and
    bypass the grounding gate; non-scalars run the gate as concept-level
    claims.  Facts already marked ``synthetic=True`` are passed through
    untouched (they bypass the gate via the legacy synthetic path).
    """
    scalar: list[dict] = []
    non_scalar: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            non_scalar.append(f)
            continue
        if f.get("synthetic") is True:
            non_scalar.append(f)
            continue
        if _is_scalar_value(str(f.get("object", ""))):
            scalar.append(f)
        else:
            non_scalar.append(f)
    return scalar, non_scalar


def _project_scalar_facts_to_attributes(graph: SessionGraph, scalar_facts: list[dict]) -> None:
    """Fold scalar-object facts onto ``Entity.attributes`` of the subject.

    Mutates ``graph.entities`` in place: ensures every scalar-fact subject
    has an ``Entity`` record (creating one with type ``"concept"`` when
    missing), then sets ``entity.attributes[<key>] = object``.  ``<key>``
    is the relation predicate with a leading ``has_`` stripped so the QA
    generator's :func:`_flatten_entity_attributes` (which re-prefixes
    attribute keys with ``has_`` when minting projected relations) does
    not produce ``has_has_<key>``.
    """
    if not scalar_facts:
        return
    name_to_entity = {e.name: e for e in graph.entities}
    for f in scalar_facts:
        subj = str(f.get("subject", "")).strip()
        pred = str(f.get("predicate", "")).strip()
        obj = str(f.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        ent = name_to_entity.get(subj)
        if ent is None:
            ent = Entity(name=subj, entity_type="concept")
            graph.entities.append(ent)
            name_to_entity[subj] = ent
        attr_key = pred.removeprefix("has_") if pred.startswith("has_") else pred
        ent.attributes[attr_key] = obj


def _entity_is_grounded(entity: str, transcript_norm: str, known_names: set[str]) -> bool:
    """True iff every significant token in `entity` appears in the transcript.

    An entity is "grounded" when:
    - It is a known real-name from the anonymization mapping (mapping.keys()), OR
    - Every significant token (>2 alphabetic chars) appears as a whole word in
      the normalised transcript.

    Entities that pass the first check are users/places explicitly mentioned
    in the session (covered by the mapping). Entities that pass only the
    second check are concepts grounded in the transcript content. Entities
    that pass neither (e.g. `CIA` inferred from a transcript that only
    mentions "Langley") are rejected — SOTA brought them in via world
    knowledge, not the user's conversation.
    """
    if not entity:
        return False
    if entity in known_names:
        return True
    norm = _normalize_for_grounding(entity).strip()
    if not norm:
        return False
    tokens = _extract_alpha_tokens(norm, min_len=3)
    if not tokens:
        # Too-short entity (e.g. initials, CJK short names like "Li Na").
        # Whole-phrase word-boundary match prevents "Li" from matching
        # inside "Libya".
        return _contains_whole_word(transcript_norm, norm)
    # Strict "all significant tokens must appear" is intentional: partial
    # matches would let SOTA world-knowledge inferences slip through (entity
    # "Munich Airport" against transcript saying only "Munich"). We favour
    # precision (drop plausibly-enriched entities) over recall.
    return all(_contains_whole_word(transcript_norm, t) for t in tokens)


_VALID_ROLE_AWARE_MODES = ("off", "diagnostic", "active")


def _apply_grounding_gate(
    facts: list[dict],
    transcript: str,
    known_names: set[str],
    *,
    speaker_name: str | None = None,
    mode: str = "off",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run the grounding gate per the configured ``role_aware_grounding`` mode.

    Returns ``(kept, dropped, would_drop_role_aware)``:

    * ``kept`` — facts to persist.  In ``off`` and ``diagnostic`` modes
      this is the role-blind result (today's behaviour).  In ``active``
      mode this is the role-aware result.
    * ``dropped`` — facts the gate dropped (always honoured by callers).
    * ``would_drop_role_aware`` — facts the role-aware gate would
      *additionally* drop versus the role-blind gate.  Populated only in
      ``diagnostic`` mode; empty in the other two modes.  Surfaces the
      structural-defense observability without changing production
      behaviour.

    ``mode`` accepts ``off`` / ``diagnostic`` / ``active``.  Unknown
    values fall back to ``off`` with a warning — production paths
    can't reach that branch because the server config validates the
    field at construction.

    ``speaker_name`` is required for ``diagnostic`` and ``active`` modes
    (it identifies whose turns count as ``[user]``).  When ``None`` the
    gate falls back to role-blind regardless of mode.

    Facts with ``synthetic=True`` bypass grounding entirely (their objects
    are pipeline-emitted scalars — emails, phones, URLs — not transcript
    spans).
    """
    if mode not in _VALID_ROLE_AWARE_MODES:
        logger.warning("Unknown role_aware_grounding mode %r; falling back to 'off'", mode)
        mode = "off"

    # Strict ``is True`` — no accidental bypass via stringly-typed flags.
    synthetic_kept = [f for f in facts if f.get("synthetic") is True]
    facts = [f for f in facts if f.get("synthetic") is not True]

    speaker_names = {speaker_name} if speaker_name else None
    if not speaker_names:
        # No speaker to anchor [user] spans against.  Behave as today.
        kept, dropped = _drop_ungrounded_facts(facts, transcript, known_names)
        return synthetic_kept + kept, dropped, []

    if mode == "active":
        kept, dropped = _drop_ungrounded_facts(
            facts, transcript, known_names, speaker_names=speaker_names
        )
        return synthetic_kept + kept, dropped, []

    # off or diagnostic: production result is the role-blind one.
    blind_kept, blind_dropped = _drop_ungrounded_facts(facts, transcript, known_names)

    if mode != "diagnostic":
        return synthetic_kept + blind_kept, blind_dropped, []

    # Diagnostic: also run role-aware to compute *additional* would-drops.
    # Identity comparison is safe — _drop_ungrounded_facts doesn't copy facts.
    _aware_kept, aware_dropped = _drop_ungrounded_facts(
        facts, transcript, known_names, speaker_names=speaker_names
    )
    blind_kept_ids = {id(f) for f in blind_kept}
    would_drop = [f for f in aware_dropped if id(f) in blind_kept_ids]
    if would_drop:
        logger.info(
            "Role-aware grounding (diagnostic): would drop %d additional fact(s) "
            "for speaker=%r; kept-by-prod=%d, dropped-by-prod=%d.",
            len(would_drop),
            speaker_name,
            len(blind_kept),
            len(blind_dropped),
        )
    return synthetic_kept + blind_kept, blind_dropped, would_drop


def _drop_ungrounded_facts(
    facts: list[dict],
    original_transcript: str,
    known_names: set[str],
    *,
    speaker_names: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Gate every fact's subject and object against transcript grounding.

    Returns ``(kept_facts, dropped_facts)``.  Dropped facts had at least
    one endpoint that was neither a known real-name nor grounded in the
    transcript — typically SOTA world-knowledge inferences rather than
    facts about the user's session.

    Two grounding modes:

    * **Role-blind** (default, ``speaker_names`` is ``None`` or empty):
      every endpoint must ground in the *full* transcript or be in
      ``known_names``.  Today's behaviour, preserved for backward
      compatibility with both production callers.

    * **Role-aware** (``speaker_names`` provided): when the *subject* of
      a triple is in ``speaker_names`` (i.e. the triple makes a claim
      *about* the speaker), the *object* must additionally ground in
      the user-only spans of the transcript — text uttered in
      ``[user]`` lines, not ``[assistant]`` lines.  Closes the
      assistant-into-graph hallucination class: triples whose
      substantive content comes only from non-speaker turns are
      dropped deterministically, regardless of whether the prompt-
      level attribution rule held.  ``known_names`` exemption still
      applies on both endpoints, so cross-role spelling corrections
      (assistant fixing a user typo) still pass.

    Subject-side grounding stays role-blind even in role-aware mode:
    the speaker can be referred to by other parties (the assistant
    addressing them by name), so an ``[assistant]`` mention of the
    subject doesn't disqualify the triple — only the *content* claim
    in the object does.
    """
    transcript_norm = _normalize_for_grounding(original_transcript)
    user_norm: str | None = None
    if speaker_names:
        # Production format from SessionBuffer._format_turns renders user turns as
        # "{speaker_name}: {text}". Build the union of user spans across every
        # configured speaker name; sorted iteration keeps user_norm reproducible
        # regardless of set ordering.
        spans: list[str] = []
        for sn in sorted(s for s in speaker_names if s):
            spans.append(_extract_user_spans(original_transcript, speaker_name=sn))
        user_norm = _normalize_for_grounding("\n".join(spans))

    kept: list[dict] = []
    dropped: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        subj = str(f.get("subject", ""))
        obj = str(f.get("object", ""))

        # Subject grounding: full transcript (subject can be referred to by
        # any participant; what matters is that they exist in the session).
        if not _entity_is_grounded(subj, transcript_norm, known_names):
            dropped.append(f)
            continue

        # Object grounding: role-aware when speaker is the subject.
        if speaker_names and user_norm is not None and subj in speaker_names:
            if not _entity_is_grounded(obj, user_norm, known_names):
                dropped.append(f)
                continue
        else:
            if not _entity_is_grounded(obj, transcript_norm, known_names):
                dropped.append(f)
                continue

        kept.append(f)
    return kept, dropped


def _apply_bindings(
    facts: list[dict],
    mapping: dict[str, str],
    sota_bindings: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """De-anonymize facts via state-machine substitution.

    Combines two substitution sources into one pass:

    * **Anonymizer mapping** (``mapping`` arg) — ``real_name -> placeholder``
      from the local anonymizer pass. Reversed here; substituted via word-
      boundary token walk so apostrophes / punctuation around bare tokens
      don't break (``Person_2's cousin`` -> ``Alex's cousin``).
    * **SOTA bindings** (``sota_bindings`` arg) — ``placeholder_name -> real_text``
      that SOTA emitted alongside its enriched facts. The braced literal
      ``{Event_1}`` is matched and replaced directly (string substitution);
      no diff against the updated transcript, no LLM call.

    Facts whose subject or object still contains a placeholder pattern after
    substitution are dropped via the existing residual sweep. Causes:
      1. SOTA introduced a braced placeholder but omitted its binding.
      2. SOTA emitted a bare placeholder that was never in the anonymizer
         mapping (anonymizer leak).
      3. Composite strings where one of multiple placeholders couldn't be
         resolved.

    Returns ``(kept_facts, dropped_facts)``.

    Replaces the previous LLM-based deanon attempt that crashed on the
    largest chunk's prompt with ``device not ready`` (VRAM exhaustion on
    Mistral 7B at 8 GiB). Also replaces the regex-based binding recovery
    (``_extract_sota_bindings``) which produced bogus mappings under
    multi-token replace blocks (bug 5).
    """
    # Map braced literal -> real_text. SOTA's contract: binding key omits braces.
    braced_map: dict[str, str] = {
        f"{{{k}}}": v
        for k, v in sota_bindings.items()
        if isinstance(k, str) and isinstance(v, str) and k and v
    }
    # Map bare placeholder -> real_name (reversed anonymizer mapping).
    bare_map: dict[str, str] = {
        ph: real
        for real, ph in mapping.items()
        if isinstance(ph, str) and isinstance(real, str) and ph and real
    }

    substituted: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        subj = str(f.get("subject", ""))
        obj = str(f.get("object", ""))
        # Step 1: substitute braced SOTA placeholders (literal string match —
        # the braces make these unambiguous, no word-boundary needed).
        for braced, real in braced_map.items():
            if braced in subj:
                subj = subj.replace(braced, real)
            if braced in obj:
                obj = obj.replace(braced, real)
        # Step 2: substitute bare anonymizer placeholders (word-boundary so
        # apostrophes and surrounding punctuation work).
        subj = _substitute_whole_words(subj, bare_map)
        obj = _substitute_whole_words(obj, bare_map)
        substituted.append({**f, "subject": subj, "object": obj})

    # Residual sweep catches unresolved placeholders (missing binding, anon
    # leak, etc.). Reuses existing tested helper.
    return _strip_residual_placeholders(substituted)


def _strip_residual_placeholders(
    facts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Drop facts whose subject or object contains a residual placeholder token.

    Runs post de-anonymization. Catches anything shaped like a placeholder —
    either braced `{Prefix_N}` or bare `Prefix_N` with capitalised prefix.
    No prefix enumeration; the pattern is type-agnostic. Covers:
    1. SOTA invented a placeholder that was never in the mapping.
    2. De-anonymization couldn't reverse-map a placeholder (mapping gap).
    3. Composite strings like `Person_2's Support` where the placeholder is
       embedded in a longer phrase (substring search).

    Returns `(kept_facts, dropped_facts)`. Each dropped fact is the exact
    input object the caller can inspect for audit / diagnostics — no
    `id()`-based reconstruction required.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        s = str(f.get("subject", ""))
        o = str(f.get("object", ""))
        if _PLACEHOLDER_TOKEN_RE.search(s) or _PLACEHOLDER_TOKEN_RE.search(o):
            dropped.append(f)
            continue
        kept.append(f)
    return kept, dropped


def _normalize_anonymization_mapping(mapping: dict) -> tuple[dict, dict]:
    """Normalize mapping to canonical {real_name: placeholder} direction.

    Per-entry classification — each (k, v) pair is placed in canonical form
    based on which side matches the placeholder regex:
    - key matches placeholder and value does not ⇒ invert this pair.
    - value matches placeholder and key does not ⇒ keep as-is.
    - both or neither match ⇒ ambiguous; drop (logging).

    Returns `(canonical_mapping, stats)` where stats has `{inverted, dropped}`
    counts — surfaces the mapping-quality signal to callers so they can
    persist it in diagnostics (ambiguous-drop can otherwise silently void
    real entities).
    """
    if not mapping:
        return mapping, {"inverted": 0, "dropped": 0}
    _pat = anonymizer_placeholder_pattern()
    if _pat is None:
        logger.warning(
            "Anonymization mapping: no anonymizer prefixes configured; "
            "dropping %d pairs — nothing can be canonicalized.",
            len(mapping),
        )
        return {}, {"inverted": 0, "dropped": len(mapping)}
    out: dict = {}
    inverted = 0
    dropped = 0
    for k, v in mapping.items():
        k_match = bool(_pat.match(str(k)))
        v_match = bool(_pat.match(str(v)))
        if k_match and not v_match:
            out[v] = k
            inverted += 1
        elif v_match and not k_match:
            out[k] = v
        else:
            # Both sides match (e.g. {"Person_1": "City_1"}) or neither does —
            # we cannot tell which side is real. Dropping the pair is safer
            # than keeping it: retaining would corrupt reverse-lookup
            # (placeholder → placeholder) and silently drop facts via the
            # residual sweep with no explicit error.
            dropped += 1
    if inverted:
        logger.info(
            "Anonymization mapping: inverted %d/%d pairs to canonical "
            "{real: placeholder} direction",
            inverted,
            len(mapping),
        )
    if dropped:
        logger.warning(
            "Anonymization mapping: dropped %d/%d ambiguous pairs (both or "
            "neither side matches placeholder pattern); affected entities will "
            "not de-anonymize and their triples will be swept.",
            dropped,
            len(mapping),
        )
    return out, {"inverted": inverted, "dropped": dropped}


def _mapping_is_canonical(mapping: dict) -> bool:
    """True iff mapping is {real_name: placeholder} with all values matching.

    When no anonymizer prefixes are configured (``anonymizer_placeholder_pattern``
    returns ``None``), an empty mapping is considered canonical (no entries to
    validate), and any non-empty mapping is non-canonical (no placeholder
    vocabulary means no real-name→placeholder mapping can be valid).
    """
    if not mapping:
        return True
    _pat = anonymizer_placeholder_pattern()
    if _pat is None:
        # No placeholder vocabulary — non-empty mapping cannot be canonical.
        return not mapping
    return all(_pat.match(str(v)) for v in mapping.values()) and not any(
        _pat.match(str(k)) for k in mapping.keys()
    )


# spaCy entity label → internal type name.
#
# Coverage is the set of spaCy ``en_core_web_sm`` labels that *can*
# carry identifying information; the operator picks which subset to
# actually anonymize via ``sanitization.cloud_scope``.  Numeric and
# temporal labels (DATE, TIME, MONEY, ORDINAL, CARDINAL, PERCENT,
# QUANTITY) are intentionally excluded: they don't carry PII even
# under maximally-strict policy.
#
# Internal type names match the rest of the codebase's vocabulary
# (``Entity.entity_type``, schema.yaml prefixes) so the same names
# appear in extraction output, anonymizer placeholders, and the
# ``cloud_scope`` config — operators don't have to learn spaCy's
# label conventions to configure egress.
_SPACY_PII_LABELS = {
    "PERSON": "person",
    "GPE": "place",
    "LOC": "place",
    "ORG": "organization",
    "PRODUCT": "product",
    "FAC": "facility",
    "NORP": "group",
    "EVENT": "event",
    "WORK_OF_ART": "work",
    "LAW": "law",
    "LANGUAGE": "language",
}

# Primitive-layer default scope for ``verify_anonymization_completeness``
# and ``extract_pii_names_with_ner`` when no explicit ``pii_scope`` is
# passed.  Preserves the historical hardcoded ``{person, place}`` scope
# of these primitives so consolidation (``_sota_pipeline``) and any
# direct callers keep the prior leak-detection coverage.
#
# This is *not* the cloud-egress policy default — that is
# :data:`_CLOUD_EGRESS_DEFAULT_SCOPE`, narrower and configurable via
# ``SanitizationConfig.cloud_scope`` — different concern, different
# default.  The primitive has no policy opinion; the helper does.
_DEFAULT_PII_SCOPE: frozenset[str] = frozenset({"person", "place"})

# Cloud-egress helper default scope for
# :func:`extract_and_anonymize_for_cloud` when no explicit
# ``pii_scope`` is passed.  Mirrors the production default in
# ``SanitizationConfig.cloud_scope``; kept in sync by hand because the
# extractor module shouldn't import from server config (would invert
# the dependency direction).  Operators override at runtime via the
# config knob; this is the in-code fallback only.
_CLOUD_EGRESS_DEFAULT_SCOPE: frozenset[str] = frozenset({"person"})

# Cached per-(lang) spaCy pipelines so we don't reload on every call.
_SPACY_MODELS: dict[str, object] = {}


def _clean_ner_span(text: str) -> str:
    """Normalize a raw spaCy NER span — strip dialogue tails, possessives.

    Dialogue-format transcripts often cause spaCy to extend PERSON spans
    into the following response token (e.g. ``"Li Ming: True"`` — person
    is ``"Li Ming"``, the rest is dialogue cruft that would inflate
    false "missing mapping" flags).  ``_strip_ner_dialogue_tail`` and
    ``_strip_ner_possessive`` apply the same shape as the previous
    ``_NER_DIALOGUE_TAIL_RE`` / ``_NER_POSSESSIVE_RE`` patterns without
    regex on user-content text.
    """
    cleaned = text.strip()
    cleaned = _strip_ner_dialogue_tail(cleaned)
    cleaned = _strip_ner_possessive(cleaned)
    cleaned = cleaned.rstrip(":,.;!? ")
    return cleaned.strip()


def extract_pii_names_with_ner(
    transcript: str,
    spacy_model: str = "en_core_web_sm",
    pii_scope: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    """Independent PII detection via spaCy NER (optional defense-in-depth).

    Returns a ``{name: pii_type}`` mapping over names whose internal
    type (per :data:`_SPACY_PII_LABELS`) is in ``pii_scope``.  When
    ``pii_scope`` is ``None`` the module default :data:`_DEFAULT_PII_SCOPE`
    applies.  An empty scope yields an empty dict — operator opt-out,
    not an error.  Empty dict also returned on failure (spaCy not
    installed, model missing, etc.).

    The type information is load-bearing for the repair path: when
    extraction emits a name only as a relation participant (not as a
    typed entity), repair allocates a placeholder of the wrong category
    (e.g. ``Berlin → Person_4`` instead of ``Berlin → City_1``) unless
    NER's type is consulted.  Returning the type alongside the name
    keeps the repair correct for places-emitted-as-relation-objects.

    On a name collision between the extractor and NER (same name,
    different types) the extractor wins downstream — repair only
    consults NER when the extractor has no opinion.
    """
    scope = _DEFAULT_PII_SCOPE if pii_scope is None else frozenset(pii_scope)
    if not scope or not transcript:
        return {}
    try:
        import spacy
    except ImportError:
        logger.info("spaCy not installed — NER cross-check disabled")
        return {}
    nlp = _SPACY_MODELS.get(spacy_model)
    if nlp is None:
        try:
            nlp = spacy.load(spacy_model)
            _SPACY_MODELS[spacy_model] = nlp
        except Exception as e:
            logger.warning("spaCy model %r not loadable — NER disabled: %s", spacy_model, e)
            return {}
    try:
        doc = nlp(transcript)
    except Exception as e:
        logger.warning("spaCy NER call failed — NER disabled: %s", e)
        return {}
    names: dict[str, str] = {}
    for ent in doc.ents:
        pii_type = _SPACY_PII_LABELS.get(ent.label_)
        if pii_type is None or pii_type not in scope:
            continue
        cleaned = _clean_ner_span(ent.text)
        if cleaned:
            # First spaCy span wins on collisions inside one transcript.
            names.setdefault(cleaned, pii_type)
    return names


def verify_anonymization_completeness(
    graph: SessionGraph,
    mapping: dict,
    anon_facts: list[dict],
    anon_transcript: str,
    extra_pii_names: set[str] | dict[str, str] | None = None,
    pii_scope: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Forward-path privacy guard — scope-driven.

    Returns a list of real names that the anonymizer failed to handle properly.
    Empty list == safe. Non-empty list means callers MUST abort the SOTA call.

    Detects two failure modes:
    1. **Leak**: a real name still appears in anon_transcript or anon_facts
       (anonymizer didn't replace it). Privacy violation.
    2. **Missing mapping**: a real name has been replaced in the output but is
       NOT present in mapping.values(). De-anonymization will fail silently
       and produce placeholder strings in the final graph. Correctness gap.

    ``pii_scope`` is the set of internal type names the operator wants
    anonymized; defaults to :data:`_DEFAULT_PII_SCOPE`
    (``{"person", "place"}`` — the primitive-layer fallback, wider than
    the cloud-egress helper's ``{"person"}`` ship default to preserve
    consolidation's prior leak-detection coverage).  Names whose
    extractor-declared ``entity_type`` is outside the scope
    pass through verbatim — by design.  An empty scope yields an empty
    "real names" set: the privacy guard returns empty (no leaks
    possible because nothing is in scope), and the caller treats the
    cloud egress as a no-op.

    A substring check on in-scope names still catches compound cases
    like "Li Na's Support" where an in-scope name is embedded in an
    out-of-scope phrase.

    ``extra_pii_names``: names contributed by an independent NER pass
    (see :func:`extract_pii_names_with_ner`).  When passed as a ``dict``
    (the modern shape), it carries ``{name: type}`` and is filtered by
    ``pii_scope``.  When passed as a ``set`` (back-compat), all names
    are added unconditionally — caller is responsible for pre-filtering.
    """
    scope = _DEFAULT_PII_SCOPE if pii_scope is None else frozenset(pii_scope)
    if not scope:
        return []
    type_by_name = {e.name: e.entity_type for e in graph.entities}
    real_names = {e.name for e in graph.entities if e.name and e.entity_type in scope}
    # Defensive: pick up in-scope names from relation participants too.
    for r in graph.relations:
        for n in (r.subject, r.object):
            if n and type_by_name.get(n) in scope:
                real_names.add(n)
    # Add externally-sourced names.  Dict form is filtered by scope;
    # set form is added wholesale (caller must pre-filter).
    if extra_pii_names:
        if isinstance(extra_pii_names, dict):
            real_names |= {n for n, t in extra_pii_names.items() if n and t in scope}
        else:
            real_names |= {n for n in extra_pii_names if n}

    # Case-insensitive set of all mapped strings for coverage check.
    # Mapping direction is technically {real_name: placeholder}, but models
    # frequently emit {placeholder: real_name} instead (the old prompt
    # wording taught this direction). De-anonymization may be misaligned
    # but that's a separate bug; for the privacy guard we just need to know
    # whether the real name is *somewhere* in the mapping. Bidirectional
    # check: if the name appears in either keys or values, it's accounted for.
    mapped_tokens_lc = {str(k).lower() for k in mapping.keys() if k}
    mapped_tokens_lc |= {str(v).lower() for v in mapping.values() if v}

    problems: list[str] = []
    for name in real_names:
        # Case 1: Leak — real name still appears in anon outputs.
        # Word-boundary, case-insensitive match against the transcript.
        leaked_in_transcript = bool(anon_transcript) and _contains_whole_word(
            anon_transcript, name, case_insensitive=True
        )
        if leaked_in_transcript:
            problems.append(name)
            continue
        name_lc = name.lower()
        leaked_in_facts = False
        for fact in anon_facts:
            if not isinstance(fact, dict):
                continue
            subj = str(fact.get("subject", "")).lower()
            obj = str(fact.get("object", "")).lower()
            if name_lc in subj or name_lc in obj:
                leaked_in_facts = True
                break
        if leaked_in_facts:
            problems.append(name)
            continue
        # Case 2: Missing mapping — real name absent from output AND absent
        # from mapping. De-anonymization cannot recover it.
        if name_lc not in mapped_tokens_lc:
            problems.append(name)
    return problems


def _anonymize_transcript(transcript: str, mapping: dict) -> str:
    """Apply entity name → placeholder mapping to a transcript.

    Replaces all mapped entity names with their anonymized placeholders
    so SOTA can see the conversation context without identifying info.
    Word-boundary anchored — prevents "Li" from eating the "Li" prefix
    of "Li Ming" or the "Li" substring of "Beijing".  Longer keys are
    tried first internally so multi-word names preempt single-word
    prefixes.

    Defensive: local models occasionally emit ``null`` mapping entries
    (non-string keys/values); :func:`_substitute_whole_words` skips
    those rather than crashing.
    """
    if not mapping:
        return transcript
    invalid = [
        (k, v)
        for k, v in mapping.items()
        if not isinstance(k, str) or not isinstance(v, str) or not k
    ]
    for k, v in invalid:
        logger.warning("Skipping invalid anonymization entry: %r → %r", k, v)
    return _substitute_whole_words(transcript, mapping)


_DEFAULT_ANONYMIZER_MAX_TOKENS = 2048
_DEFAULT_ANONYMIZER_TEMPERATURE = 0.0


def load_anonymization_prompt() -> str:
    """Single source of truth for the anonymization prompt.

    Both the local-model and cloud-extractor anonymization paths read through
    this helper so a `configs/prompts/anonymization.txt` override applies to
    both — no silent divergence.
    """
    return _load_prompt("anonymization.txt", _DEFAULT_ANONYMIZATION_PROMPT)


def _anonymize_with_local_model(
    graph: SessionGraph,
    model,
    tokenizer,
    transcript: str = "",
    max_tokens: int = _DEFAULT_ANONYMIZER_MAX_TOKENS,
    temperature: float = _DEFAULT_ANONYMIZER_TEMPERATURE,
) -> tuple[list[dict] | None, dict, str]:
    """Anonymize extracted facts AND transcript using the local model.

    Returns (anonymized_facts, mapping, anonymized_transcript) on success or
    (None, {}, "") on failure. The mapping is total over both inputs by
    contract — every real name appearing in either facts or transcript MUST
    be a value in the mapping, so the reverse mapping is total too.

    Backward-compat: if `transcript` is empty, an empty anonymized transcript
    is returned. Existing callers (older tests) still work.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    facts = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in graph.relations
    ]

    anon_prompt = load_anonymization_prompt()
    prompt = anon_prompt.format(
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(no transcript provided)",
        replacement_rules=format_replacement_rules(),
    )
    messages = [
        {"role": "system", "content": "You anonymize data. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )

    raw = generate_answer(
        model, tokenizer, formatted, max_new_tokens=max_tokens, temperature=temperature
    )
    logger.debug("Anonymization raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        if isinstance(data, dict) and "mapping" in data:
            normalized, _ = _normalize_anonymization_mapping(data["mapping"])
            # New schema: anonymized_facts + anonymized_transcript
            if "anonymized_facts" in data:
                return (
                    data["anonymized_facts"],
                    normalized,
                    data.get("anonymized_transcript", ""),
                )
            # Backward-compat: old schema with "anonymized" key (facts only)
            if "anonymized" in data:
                return data["anonymized"], normalized, ""
        logger.warning("Anonymization returned unexpected format")
        return None, {}, ""
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Anonymization parse failed: %s", e)
        return None, {}, ""


# Public provider metadata — single source of truth, reused by callers
# (scripts/dev/compare_extraction.py and the production server) so they can dispatch
# by provider consistently with this module.
OPENAI_COMPAT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}
OPENAI_COMPAT_PROVIDERS = set(OPENAI_COMPAT_ENDPOINTS) | {"ollama"}

# Env var holding the API key for each supported provider. Extended whenever
# a new provider is added to _filter_with_sota.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

# Symmetric predicates — relations that hold equally in both directions.
# The enrichment prompt instructs the LLM to canonicalize these (emit one
# direction only, lex-ordered subject < object). This set is the deterministic
# post-processing safety net: if the LLM left both directions, drop the one
# with subject > object. Kept in sync with the list in sota_enrichment.txt.
SYMMETRIC_PREDICATES = frozenset(
    {
        "friend_of",
        "friends_with",
        "married_to",
        "spouse_of",
        "sibling_of",
        "brother_of",
        "sister_of",
        "cousin_of",
        "colleague_of",
        "coworker_of",
        "neighbor_of",
        "partner_of",
        "related_to",
        "workout_partner_of",
        "study_partner_of",
        "met_with",
        "talked_to",
        "knows",
        "agrees_with",
        "disagrees_with",
        "attended_with",
        "attends_gym_with",
        "shares_interest_with",
    }
)


def _canonicalize_symmetric_predicates(relations: list) -> list:
    """Drop redundant inverse triples for symmetric predicates.

    Deterministic safety net for the LLM-driven canonicalization in the
    enrichment prompt. For each (subject, predicate, object) where
    `predicate ∈ SYMMETRIC_PREDICATES` and the inverse triple is also
    present, keep only the one with subject ≤ object lexicographically.
    """
    if not relations:
        return relations

    # Build index of (subject, predicate, object) tuples for O(1) inverse lookup.
    seen = {(r.subject, r.predicate, r.object) for r in relations}
    kept = []
    for r in relations:
        if r.predicate in SYMMETRIC_PREDICATES and r.subject > r.object:
            inverse = (r.object, r.predicate, r.subject)
            if inverse in seen:
                # Inverse will be (or has been) kept; drop this one.
                logger.debug(
                    "Symmetric dedup: dropped %s --[%s]--> %s (inverse kept)",
                    r.subject,
                    r.predicate,
                    r.object,
                )
                continue
        kept.append(r)
    return kept


_SOTA_ENRICHMENT_SYSTEM_PROMPT = (
    "You are a knowledge graph enrichment assistant. "
    "Resolve coreference and split compound facts. Do NOT remove facts — a "
    "separate plausibility filter handles removal. Output valid JSON only."
)
_SOTA_PLAUSIBILITY_SYSTEM_PROMPT = (
    "You are a knowledge graph plausibility filter. "
    "Drop invalid facts only. Do NOT add or modify facts. Output valid JSON only."
)
# Backward-compatible alias for any external caller of the old name.
_SOTA_SYSTEM_PROMPT = _SOTA_ENRICHMENT_SYSTEM_PROMPT


# _DEFAULT_FILTER_MAX_TOKENS / _DEFAULT_FILTER_TEMPERATURE /
# _DEFAULT_FILTER_TIMEOUT_SECONDS are defined at the top of this module —
# they need to precede the extract_graph signature that references them.


def _filter_anthropic(
    prompt: str,
    api_key: str,
    filter_model: str,
    system_prompt: str = _SOTA_ENRICHMENT_SYSTEM_PROMPT,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> str | None:
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping SOTA filter")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        response = client.messages.create(
            model=filter_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
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
    system_prompt: str = _SOTA_ENRICHMENT_SYSTEM_PROMPT,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> str | None:
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
    except (httpx.HTTPError, httpx.RequestError, KeyError, IndexError) as e:
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
    system_prompt: str = _SOTA_ENRICHMENT_SYSTEM_PROMPT,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> str | None:
    """Generic SOTA dispatch (anthropic native or any OpenAI-compatible host)."""
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
        logger.warning("SOTA response parse failed: %s", e)
        return None


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
) -> tuple[list[dict] | None, str | None, dict[str, str], str | None, dict]:
    """SOTA enrichment pass — coreference + compound splitting + safe reification.

    Returns ``(facts, updated_transcript, new_entity_bindings, raw_response, info)``.

    ``new_entity_bindings`` maps each new braced placeholder SOTA introduced
    (without braces, e.g. ``"Event_1"``) to the exact transcript span it stands
    for. Empty dict when SOTA introduced no new entities or used the legacy
    response shape that lacked the field. SOTA already knows the binding at
    the moment it mints each placeholder, so emitting it explicitly removes
    the entire transcript-diff reconstruction problem the previous pipeline
    used to do post-hoc.

    ``info`` is a dict with diagnostic flags the caller persists into
    ``graph.diagnostics``:

    - ``parse_path``: ``"preferred"`` (strict JSON, dict with ``facts`` + ``updated_transcript``)
      or ``"legacy_fallback"`` (response failed strict parse; bare-array salvage).
    - ``response_truncated``: ``True`` when strict parse failed (typically max_tokens
      hit; JSON ends mid-string). Salvaged facts may be a strict subset of what
      the model intended to emit — a silent kill site historically.
    - ``response_chars``: length of the raw response in characters, for sizing
      decisions and budget tracking.
    - ``preferred_fact_count`` / ``legacy_fact_count``: count after each parse path,
      whichever ran. Lets us measure salvage rate when truncation hits.
    - ``new_entity_bindings_count``: number of SOTA-introduced placeholders for
      which the response carried an explicit binding. Useful to spot when the
      model omits the field (legacy-shape responses surface as 0).

    Legacy responses (bare JSON array, or dict missing ``new_entity_bindings``)
    are accepted — in that case bindings is ``{}`` and any braced placeholders
    in the facts will fail substitution downstream and the corresponding facts
    will be dropped by the residual sweep / grounding gate.
    """
    enrichment_prompt = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
    prompt = enrichment_prompt.format(
        facts_json=json.dumps(anon_facts, indent=2),
        transcript=anon_transcript or "(not available)",
    )
    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=_SOTA_ENRICHMENT_SYSTEM_PROMPT,
        timeout_seconds=timeout_seconds,
    )
    if raw is None:
        return None, None, {}, None, {"parse_path": "no_response"}
    info: dict = {"response_chars": len(raw)}
    # Preferred schema:
    # {"facts": [...], "updated_transcript": "...", "new_entity_bindings": {...}}
    try:
        parsed = json.loads(_extract_json_block(raw))
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            bindings_raw = parsed.get("new_entity_bindings") or {}
            bindings: dict[str, str] = {}
            if isinstance(bindings_raw, dict):
                bindings = {
                    str(k): str(v)
                    for k, v in bindings_raw.items()
                    if isinstance(k, str) and isinstance(v, str) and k and v
                }
            info["parse_path"] = "preferred"
            info["response_truncated"] = False
            info["preferred_fact_count"] = len(parsed["facts"])
            info["new_entity_bindings_count"] = len(bindings)
            return parsed["facts"], parsed.get("updated_transcript"), bindings, raw, info
    except (json.JSONDecodeError, ValueError):
        pass
    # Legacy / salvage: strict parse failed. Most common cause is max_tokens
    # truncation cutting JSON mid-string. Best-effort fact recovery only.
    salvaged = _parse_facts_response(raw)
    info["parse_path"] = "legacy_fallback"
    info["response_truncated"] = True
    info["legacy_fact_count"] = len(salvaged) if salvaged else 0
    info["new_entity_bindings_count"] = 0
    logger.warning(
        "SOTA strict-parse failed; salvaged %d fact(s) via legacy fallback "
        "(response_chars=%d, max_tokens=%d) — likely truncation",
        info["legacy_fact_count"],
        info["response_chars"],
        max_tokens,
    )
    return salvaged, None, {}, raw, info


# ---------------------------------------------------------------------------
# Graph-level SOTA enrichment (Task #10)
# ---------------------------------------------------------------------------

_SOTA_GRAPH_ENRICHMENT_SYSTEM_PROMPT = (
    "You are a knowledge graph enrichment assistant operating over a pre-merged "
    "cross-transcript graph. Emit cross-session second-order relations and same_as "
    "pairs for duplicate entities. Output valid JSON only."
)

_DEFAULT_GRAPH_ENRICHMENT_PROMPT = """\
You are operating over a pre-merged, cross-transcript knowledge graph.
Identify second-order cross-session relations implied by the input triples,
and same_as pairs for nodes that refer to the same real-world entity.

Rules:
- Only emit relations with confidence >= 0.7.
- Do NOT emit relations already in the input.
- For symmetric predicates emit only ONE direction (subject < object, lexicographically).
- relation_type must be one of: factual, temporal, preference, social.
- subject and object must be node names from the input graph.

Input triples (JSON):
{triples_json}

Return ONLY valid JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...",
"relation_type": "...", "confidence": 0.0}}], "same_as": [["canonical", "variant"]]}}
"""


def _graph_enrich_with_sota(
    triples: list[dict],
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
    timeout_seconds: float = _DEFAULT_FILTER_TIMEOUT_SECONDS,
) -> tuple[list[dict], list[list[str]], str | None] | None:
    """SOTA graph-level enrichment pass over a pre-merged cumulative graph.

    Sends a subgraph serialized as triples to a SOTA provider and requests
    two outputs:
    - New cross-session second-order relations not already in the graph.
    - ``same_as`` pairs identifying duplicate nodes under different surface forms.

    Loads ``sota_graph_enrichment.txt`` with ``_DEFAULT_GRAPH_ENRICHMENT_PROMPT``
    as fallback. The prompt uses a ``{triples_json}`` placeholder.

    Args:
        triples: List of ``{"subject", "predicate", "object", "relation_type"}``
            dicts representing the chunk subgraph.
        api_key: Provider API key.
        provider: SOTA provider name (e.g. ``"anthropic"``).
        filter_model: Model identifier for the provider.
        endpoint: Custom endpoint for OpenAI-compatible providers.
        max_tokens: Maximum tokens in the SOTA response.
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        ``(new_relations, same_as_pairs, raw_response)`` on success, or
        ``None`` when the SOTA call fails or the response cannot be parsed.
        ``new_relations`` is a list of relation dicts; ``same_as_pairs`` is a
        list of ``[canonical, variant]`` pairs.
    """
    enrichment_prompt = _load_prompt("sota_graph_enrichment.txt", _DEFAULT_GRAPH_ENRICHMENT_PROMPT)
    try:
        prompt = enrichment_prompt.format(triples_json=json.dumps(triples, indent=2))
    except KeyError as exc:
        logger.warning("Graph enrichment prompt has unexpected placeholder: %s", exc)
        return None

    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=_SOTA_GRAPH_ENRICHMENT_SYSTEM_PROMPT,
        timeout_seconds=timeout_seconds,
    )
    if raw is None:
        return None

    # Parse response: preferred schema {"relations": [...], "same_as": [...]}
    try:
        json_str = _extract_json_block(raw)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Graph enrichment response parse failed: %s", exc)
        return None

    if isinstance(parsed, list):
        # Legacy bare-array: treat as relations, no same_as.
        logger.debug("Graph enrichment: bare-array response (no same_as)")
        return parsed, [], raw

    if not isinstance(parsed, dict):
        logger.warning("Graph enrichment: unexpected response type %s", type(parsed).__name__)
        return None

    new_relations: list[dict] = parsed.get("relations") or []
    raw_same_as = parsed.get("same_as") or []

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

    if not isinstance(new_relations, list):
        logger.warning("Graph enrichment: 'relations' is not a list, ignoring")
        new_relations = []

    return new_relations, same_as_pairs, raw


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
) -> tuple[list[dict] | None, str | None]:
    """SOTA plausibility filter — drops invalid relations only.

    No additions, no modifications. See sota_plausibility.txt for the
    drop criteria (self-loops, tautologies, role leaks, etc.).

    Returns `(facts, raw_response)`. Raw response is preserved so callers
    can inspect the judge's verdict when questioning drop decisions.
    """
    plaus_prompt = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
    prompt = plaus_prompt.format(
        facts_json=json.dumps(enriched_anon_facts, indent=2),
        transcript=anon_transcript or "(not available)",
    )
    raw = _sota_call(
        prompt,
        api_key,
        provider,
        filter_model,
        endpoint,
        max_tokens,
        temperature,
        system_prompt=_SOTA_PLAUSIBILITY_SYSTEM_PROMPT,
        timeout_seconds=timeout_seconds,
    )
    return _parse_facts_response(raw, strict_array=True), raw


def _local_plausibility_filter(
    facts: list[dict],
    transcript: str,
    model,
    tokenizer,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
) -> list[dict] | None:
    """Local-model plausibility filter — drops invalid relations only.

    Same prompt as the SOTA plausibility filter, executed by a local model.
    Caller decides what data to pass: anonymized facts (placeholder strings)
    or de-anonymized facts (real names). The prompt is stage-agnostic.

    Returns the filtered list or None on parse failure (caller falls back).
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    _vram_snapshot(f"plaus_filter_entry n_facts={len(facts)}")
    plaus_prompt = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
    prompt = plaus_prompt.format(
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(not available)",
    )
    messages = [
        {"role": "system", "content": _SOTA_PLAUSIBILITY_SYSTEM_PROMPT},
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
    try:
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
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
    return _parse_facts_response(raw, strict_array=True)
