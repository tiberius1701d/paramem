"""LLM-based knowledge graph extraction — generate once, parse once."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from paramem.graph.schema import SessionGraph
from paramem.graph.schema_config import (
    anonymizer_placeholder_pattern,
    anonymizer_prefix_to_type,
    anonymizer_type_to_prefix,
    entity_types,
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


# ---------------------------------------------------------------------------
# Word-boundary substitution / matching helpers — replace fragile
# ``re.sub(rf"\b{re.escape(name)}\b", ...)`` patterns on user-content text
# with structural token walks.  The bounded internal placeholder grammars
# (``_PLACEHOLDER_TOKEN_RE`` / ``_BRACED_PLACEHOLDER_RE`` etc.) keep their
# regex form because they match strings we mint ourselves; these helpers
# replace the regex sites that operate on free-form user transcripts /
# LLM output, where the project rule rejects regex.
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
        graph = SessionGraph.model_validate(data)
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
    temperature: float = 0.0,
    max_tokens: int = 2048,
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
        graph = _parse_extraction(raw_output, session_id)
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
            role_aware_grounding=role_aware_grounding,
            pii_scope=pii_scope,
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


def _parse_extraction(raw_output: str, session_id: str) -> SessionGraph:
    """Parse raw model output into a SessionGraph.

    Handles non-standard field names, array-valued fields, and other
    model output quirks via _normalize_extraction.
    """
    json_str = _extract_json_block(raw_output)
    data = json.loads(json_str)

    data["session_id"] = session_id
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    data = _normalize_extraction(data)

    graph = SessionGraph.model_validate(data)
    logger.info(
        "Extracted graph: %d entities, %d relations (session=%s)",
        len(graph.entities),
        len(graph.relations),
        session_id,
    )
    return graph


def _extract_json_block(text: str) -> str:
    """Extract JSON from model output, handling markdown code blocks."""
    # Try markdown code blocks
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            # Find closing ``` — fall through to brace matching if missing
            closing = text.find("```", start)
            if closing != -1:
                return text[start:closing].strip()

    # Fall back to finding raw JSON by brace/bracket matching
    # Try object first, then array
    brace_start = text.find("{")
    bracket_start = text.find("[")

    # Use whichever comes first
    if brace_start == -1 and bracket_start == -1:
        raise ValueError("No JSON found in model output")

    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        # Array — match [ ]
        open_char, close_char = "[", "]"
        start = bracket_start
    else:
        # Object — match { }
        open_char, close_char = "{", "}"
        start = brace_start

    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced braces in model output")


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
            norm["entity_type"] = raw_type if raw_type in set(entity_types()) else fb_etype
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
    role_aware_grounding: str = "off",
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
            max_tokens=_DEFAULT_FILTER_MAX_TOKENS,
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
    anon_facts, mapping, anon_transcript = _anonymize_with_local_model(
        graph, model, tokenizer, transcript=transcript
    )
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
            role_aware_grounding=role_aware_grounding,
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
    # Fall back to mechanical replacement if the model didn't return an
    # anonymized transcript (older anonymization prompt or partial response).
    if not anon_transcript:
        anon_transcript = _anonymize_transcript(transcript, mapping)

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
                    role_aware_grounding=role_aware_grounding,
                )

    # Step 2: SOTA enrichment — coreference + compound splitting + safe reification.
    # Skipped when _skip_sota=True (residual leak after repair with canonical mapping).
    updated_anon_transcript = None
    _sota_raw = None
    if _skip_sota:
        # Skip SOTA — use filtered anon_facts as-is.
        enriched_anon = anon_facts
        logger.info(
            "Skipping SOTA enrichment (residual leak path); using %d fact(s)", len(anon_facts)
        )
    else:
        # Brace every known placeholder at the SOTA boundary so the enricher sees a
        # consistent `{Prefix_N}` convention throughout its input. SOTA continues the
        # pattern for any new placeholders it introduces; we diff updated transcript
        # vs input to recover their bindings.
        known_placeholders = set(mapping.values())
        braced_transcript = _brace_placeholders_in_text(anon_transcript, known_placeholders)
        braced_facts = _brace_placeholders_in_facts(anon_facts, known_placeholders)
        enriched_anon, updated_anon_transcript, _sota_raw, _sota_info = _filter_with_sota(
            braced_facts, api_key, provider, filter_model, braced_transcript, endpoint=endpoint
        )
        # Persist SOTA-call telemetry: parse path, truncation flag, response size,
        # salvage count. Lets us see truncation kill sites that were previously
        # silent (strict-parse failure → legacy salvage of an unknown subset).
        if _sota_info:
            graph.diagnostics["sota_call_info"] = _sota_info
        anon_transcript = braced_transcript  # bindings diff is against what we sent
        if enriched_anon is None:
            logger.warning("SOTA enrichment failed — keeping pre-enrichment facts")
            enriched_anon = anon_facts

        if not enriched_anon:
            logger.info("SOTA enrichment removed all relations")

        # Step 2b: recover bindings for SOTA-introduced placeholders via transcript diff.
        # SOTA must insert braced tokens (e.g. `{Event_1}`) in the updated transcript at
        # the position of the span they represent. Ungrounded placeholders (those in
        # facts but not in the updated transcript) are caught by the residual sweep below.
        if updated_anon_transcript:
            bindings = _extract_sota_bindings(anon_transcript, updated_anon_transcript)
            for real_span, placeholder in bindings.items():
                mapping.setdefault(real_span, placeholder)
            if bindings:
                logger.info("SOTA introduced %d new binding(s) via transcript diff", len(bindings))
        # Strip braces from subject/object fields so downstream lookups use bare tokens.
        enriched_anon = _strip_placeholder_braces(enriched_anon)

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
                max_tokens=_DEFAULT_FILTER_MAX_TOKENS,
                temperature=_DEFAULT_FILTER_TEMPERATURE,
            )
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

    # Step 3b: De-anonymize — reverse the mapping for all surviving/new relations.
    # Extended mapping from repair + SOTA bindings is already included in `mapping`.
    reverse_mapping = {v: k for k, v in mapping.items()}
    from paramem.graph.schema import Entity, Relation

    # Substring replacement via _substitute_whole_words: word-boundary
    # anchored, longest-placeholder-first internally to prevent partial
    # matches.  Handles composite strings like "Person_2's cousin" →
    # "David's cousin" — the apostrophe is not a word character so the
    # placeholder match terminates correctly at "Person_2".
    _reverse_mapping = {
        k: v for k, v in reverse_mapping.items() if isinstance(k, str) and isinstance(v, str) and k
    }

    def _deanonymize_field(value: str) -> str:
        return _substitute_whole_words(value, _reverse_mapping)

    deanon_facts = []
    for fact in enriched_anon:
        if not isinstance(fact, dict):
            continue
        deanon_facts.append(
            {
                **fact,
                "subject": _deanonymize_field(fact.get("subject", "")),
                "object": _deanonymize_field(fact.get("object", "")),
            }
        )

    # Step 3c: Deterministic placeholder sweep — drop any fact where SOTA invented a
    # placeholder or de-anon couldn't resolve one. Either case would ship a
    # literal "Person_3" string into the adapter.
    deanon_facts, dropped_facts = _strip_residual_placeholders(deanon_facts)
    if dropped_facts:
        graph.diagnostics["residual_dropped_facts"] = dropped_facts
        logger.warning(
            "Dropped %d fact(s) with residual placeholder strings post-de-anon.",
            len(dropped_facts),
        )

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
        filtered_deanon = _local_plausibility_filter(
            deanon_facts,
            transcript,  # original real-name transcript — intentional, see docstring
            model,
            tokenizer,
            max_tokens=_DEFAULT_FILTER_MAX_TOKENS,
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
            role_aware_grounding=role_aware_grounding,
        )

    # Rebuild entity list from surviving + new relations.
    # Every relation endpoint must have a corresponding Entity record.
    # Entity type inference uses anonymizer_prefix_to_type() from configs/schema.yaml.
    # Safe fallback is "concept", never "person" — that would stamp locations,
    # media, and free-text entities as persons, as confirmed by
    # sim_20260415_160543/graph.json.
    kept_names = {r.subject for r in kept_relations} | {r.object for r in kept_relations}
    existing_names = {e.name for e in graph.entities}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    for name in kept_names - existing_names:
        # Infer entity type from anonymizer placeholder prefix when available.
        # Prefix-to-type mapping comes from anonymizer.prefixes in configs/schema.yaml
        # (via anonymizer_prefix_to_type()). Safe fallback is "concept"
        # (matches _normalize_extraction for unknown types).
        # Never default to "person" — that stamps locations, media, devices, and
        # free-text entities as persons.
        entity_type = "concept"
        placeholder = reverse_mapping.get(name)
        if placeholder:
            prefix = placeholder.split("_")[0].lower()
            entity_type = anonymizer_prefix_to_type().get(prefix, "concept")
        graph.entities.append(Entity(name=name, entity_type=entity_type))

    graph.relations = kept_relations

    added = len(kept_relations) - original_count
    logger.info(
        "SOTA enrichment: %d → %d relations (%+d)",
        original_count,
        len(kept_relations),
        added,
    )
    return graph


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
    """Concatenate the text of every ``[user]`` (or ``user:``) line.

    Production transcripts tag turns with ``[user]`` / ``[assistant]``
    line prefixes (also ``user:`` / ``assistant:`` colon form).  This
    helper isolates the user-authored content so role-aware grounding
    can verify a triple's substantive tokens come from the speaker
    themselves, not from an assistant or third-party turn.

    When ``speaker_name`` is provided, lines whose lowercased form starts
    with ``{speaker_name.lower()}:`` are also accepted as user turns.
    This covers production-format transcripts rendered by
    ``SessionBuffer._format_turns`` as ``{speaker_name}: {text}`` instead
    of the legacy ``[user]`` / ``user:`` prefix.

    Lines that don't carry a recognised role prefix are conservatively
    dropped — the goal is high precision on "user said this", not
    inclusivity.  Mirrors the asymmetric extraction in
    :func:`_correct_entity_names` (which pulls ``[assistant]`` text).
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
    for line in (transcript or "").split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("[user]"):
            prefix_len = stripped.index("]") + 1
            out.append(stripped[prefix_len:].lstrip())
        elif lower.startswith("user:"):
            out.append(stripped[len("user:") :].lstrip())
        elif speaker_prefix and lower.startswith(speaker_prefix):
            out.append(stripped[speaker_prefix_skip:].lstrip())
    return "\n".join(out)


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
    """
    if mode not in _VALID_ROLE_AWARE_MODES:
        logger.warning("Unknown role_aware_grounding mode %r; falling back to 'off'", mode)
        mode = "off"

    speaker_names = {speaker_name} if speaker_name else None
    if not speaker_names:
        # No speaker to anchor [user] spans against.  Behave as today.
        kept, dropped = _drop_ungrounded_facts(facts, transcript, known_names)
        return kept, dropped, []

    if mode == "active":
        kept, dropped = _drop_ungrounded_facts(
            facts, transcript, known_names, speaker_names=speaker_names
        )
        return kept, dropped, []

    # off or diagnostic: production result is the role-blind one.
    blind_kept, blind_dropped = _drop_ungrounded_facts(facts, transcript, known_names)

    if mode != "diagnostic":
        return blind_kept, blind_dropped, []

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
    return blind_kept, blind_dropped, would_drop


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


# Output budget for SOTA enrichment / plausibility / graph-enrich calls.
# 2048 was hit by document-mode ingestion (resume produced ~6045 chars of
# response and JSON parse failed mid-string). 8192 is provisional headroom;
# truncation should be observable, not silent — see _filter_with_sota
# diagnostics. Long-term, output budgeting + chunked invocation is the
# durable answer (see plan Phase A0).
_DEFAULT_FILTER_MAX_TOKENS = 8192
# Validator temperature: deterministic by default. Threaded all the way to the
# provider call so Anthropic and OpenAI-compatible filters match exactly.
_DEFAULT_FILTER_TEMPERATURE = 0.0


def _filter_anthropic(
    prompt: str,
    api_key: str,
    filter_model: str,
    system_prompt: str = _SOTA_ENRICHMENT_SYSTEM_PROMPT,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
) -> str | None:
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping SOTA filter")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
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
        with httpx.Client(timeout=30.0) as client:
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
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("SOTA response parse failed: %s", e)
        return None


_BRACED_PLACEHOLDER_RE = re.compile(r"\{(\w+_\d+)\}")
# Match a bare placeholder token not already wrapped in braces (defensive: lets
# _brace_placeholders_in_text be idempotent even on accidentally-pre-braced input).
_BARE_PLACEHOLDER_RE = re.compile(r"(?<!\{)\b(\w+_\d+)\b(?!\})")


def _brace_placeholders_in_text(text: str, placeholders: set[str]) -> str:
    """Wrap each occurrence of known placeholder tokens in braces.

    Applied at the SOTA boundary so the enricher sees a consistent
    `{Prefix_N}` convention for every placeholder — both ours and any it
    introduces. Bare tokens not in `placeholders` are left untouched (real
    entity names with incidental digits like `user_42`).
    """
    if not text or not placeholders:
        return text

    def _wrap(m: re.Match) -> str:
        return f"{{{m.group(1)}}}" if m.group(1) in placeholders else m.group(0)

    return _BARE_PLACEHOLDER_RE.sub(_wrap, text)


def _brace_placeholders_in_facts(facts: list[dict], placeholders: set[str]) -> list[dict]:
    """Wrap subject/object placeholder tokens in braces (SOTA-facing form)."""
    out = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        copy = dict(f)
        for key in ("subject", "object"):
            val = str(copy.get(key, ""))
            if val in placeholders:
                copy[key] = f"{{{val}}}"
        out.append(copy)
    return out


def _extract_sota_bindings(old_transcript: str, new_transcript: str) -> dict[str, str]:
    """Recover SOTA-introduced placeholder bindings via token-level span diff.

    SOTA returns an updated transcript where newly-introduced entities are
    marked as braced placeholders (`{Event_1}`) replacing the real span they
    represent. This function aligns the two transcripts token-by-token and
    reconstructs `{real_span: placeholder}` bindings for each replacement
    whose new side is a single braced token.

    Returns bindings with real spans stripped of trailing punctuation.
    """
    import difflib

    old_toks = old_transcript.split()
    new_toks = new_transcript.split()
    matcher = difflib.SequenceMatcher(a=old_toks, b=new_toks)
    bindings: dict[str, str] = {}
    brace_token_re = re.compile(r"^\{(\w+_\d+)\}([.,;:!?)\]]*)$")
    # Hoist pattern compile out of the loop — recompiling per iteration is wasteful
    # and the None-vocab guard must be consistent across the entire diff pass.
    _pat = anonymizer_placeholder_pattern()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        if j2 - j1 != 1:
            continue
        m = brace_token_re.match(new_toks[j1])
        if not m:
            continue
        placeholder = m.group(1)
        span = " ".join(old_toks[i1:i2]).rstrip(".,;:!?'\")]")
        # Defensive: skip placeholder→placeholder bindings. SOTA may brace a
        # pre-existing bare placeholder (e.g. `Person_2` → `{Person_2}`) as
        # "normalisation". The diff would record `span=Person_2` → placeholder
        # `Person_2`. Adding that to the mapping (as `mapping["Person_2"] =
        # "Person_2"`) would corrupt the reverse lookup — a later pair like
        # `{Li Ming: Person_2}` gets overwritten, breaking de-anonymization.
        # Reject both braced spans and bare-placeholder spans.
        if (
            brace_token_re.match(span)
            or not span
            or (_pat is not None and _pat.match(span))
            or span == placeholder
        ):
            continue
        if placeholder not in bindings.values():
            bindings[span] = placeholder
    return bindings


def _strip_placeholder_braces(facts: list[dict]) -> list[dict]:
    """Remove enclosing braces from placeholder references anywhere in subject/object.

    Handles both exact-match (`{Event_1}` → `Event_1`) and inline occurrences
    (`attended {Event_1}` → `attended Event_1`). Downstream de-anonymization
    looks up bare placeholder tokens in the reverse mapping.
    """
    inline_brace_re = re.compile(r"\{(\w+_\d+)\}")
    out: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        copy = dict(f)
        for key in ("subject", "object"):
            val = str(copy.get(key, ""))
            copy[key] = inline_brace_re.sub(r"\1", val)
        out.append(copy)
    return out


def _filter_with_sota(
    anon_facts: list[dict],
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    anon_transcript: str | None = None,
    endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    temperature: float = _DEFAULT_FILTER_TEMPERATURE,
) -> tuple[list[dict] | None, str | None, str | None, dict]:
    """SOTA enrichment pass — coreference + compound splitting + safe reification.

    Returns ``(facts, updated_transcript, raw_response, info)``. ``info`` is
    a dict with diagnostic flags the caller persists into ``graph.diagnostics``:

    - ``parse_path``: ``"preferred"`` (strict JSON, dict with ``facts`` + ``updated_transcript``)
      or ``"legacy_fallback"`` (response failed strict parse; bare-array salvage).
    - ``response_truncated``: ``True`` when strict parse failed (typically max_tokens
      hit; JSON ends mid-string). Salvaged facts may be a strict subset of what
      the model intended to emit — a silent kill site historically.
    - ``response_chars``: length of the raw response in characters, for sizing
      decisions and budget tracking.
    - ``preferred_fact_count`` / ``legacy_fact_count``: count after each parse path,
      whichever ran. Lets us measure salvage rate when truncation hits.

    Legacy responses (bare JSON array, no transcript) are accepted — in that
    case ``updated_transcript`` is None and callers must fall back to the
    registry-based drop filter for any unresolved placeholders.
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
    )
    if raw is None:
        return None, None, None, {"parse_path": "no_response"}
    info: dict = {"response_chars": len(raw)}
    # Preferred schema: {"facts": [...], "updated_transcript": "..."}
    try:
        parsed = json.loads(_extract_json_block(raw))
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            info["parse_path"] = "preferred"
            info["response_truncated"] = False
            info["preferred_fact_count"] = len(parsed["facts"])
            return parsed["facts"], parsed.get("updated_transcript"), raw, info
    except (json.JSONDecodeError, ValueError):
        pass
    # Legacy / salvage: strict parse failed. Most common cause is max_tokens
    # truncation cutting JSON mid-string. Best-effort fact recovery only.
    salvaged = _parse_facts_response(raw)
    info["parse_path"] = "legacy_fallback"
    info["response_truncated"] = True
    info["legacy_fact_count"] = len(salvaged) if salvaged else 0
    logger.warning(
        "SOTA strict-parse failed; salvaged %d fact(s) via legacy fallback "
        "(response_chars=%d, max_tokens=%d) — likely truncation",
        info["legacy_fact_count"],
        info["response_chars"],
        max_tokens,
    )
    return salvaged, None, raw, info


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
    raw = generate_answer(
        model, tokenizer, formatted, max_new_tokens=max_tokens, temperature=temperature
    )
    return _parse_facts_response(raw, strict_array=True)
