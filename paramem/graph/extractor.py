"""LLM-based knowledge graph extraction — generate once, parse once."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from paramem.graph.schema import SessionGraph

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"

_DEFAULT_EXTRACTION_SYSTEM = "You are a precise knowledge graph extractor. Output valid JSON only."

_DEFAULT_EXTRACTION_PROMPT = """\
Extract all entities and relations from this conversation transcript.

Rules:
- Entity types: person, place, organization, concept, preference, event
- Relation types: factual, temporal, preference, social
- Use the EXACT names from the transcript — never substitute or rename entities
- Include attributes like age, role, date where mentioned
- predicate: snake_case verb phrase — consistent, reusable across sessions
- confidence: 1.0 for explicit facts, 0.7 for implied/inferred

Predicate examples (use these exact forms when they apply):
- married_to, parent_of, child_of, sibling_of, has_pet, lives_with
- lives_in, works_at, studies_at, born_in, prefers, knows
- manages, attended, visited, bought, read, fixed, cooked, watched
- If none fit, create a new snake_case predicate (e.g. "competed_in")

Transcript:
{transcript}

Extract all entities and relations as JSON."""


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
) -> tuple[str, str]:
    """Load extraction prompts from a directory, with hardcoded fallbacks.

    Args:
        prompts_dir: Directory containing extraction_system.txt and extraction.txt.
                     Falls back to configs/prompts/ in the project root, then to
                     hardcoded defaults.

    Returns:
        (system_prompt, extraction_prompt) tuple.
    """
    pd = Path(prompts_dir) if prompts_dir else None
    system = _load_prompt("extraction_system.txt", _DEFAULT_EXTRACTION_SYSTEM, pd)
    prompt = _load_prompt("extraction.txt", _DEFAULT_EXTRACTION_PROMPT, pd)
    return system, prompt


_DEFAULT_PROCEDURAL_PROMPT = """\
Extract preferences, habits, and routines from this conversation.
Only extract relation_type "preference". Return JSON.

Transcript:
{transcript}

Return JSON with entities, relations, summary.
"""


def load_procedural_prompt(
    prompts_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Load procedural extraction prompts."""
    pd = Path(prompts_dir) if prompts_dir else None
    system = _load_prompt("extraction_system.txt", _DEFAULT_EXTRACTION_SYSTEM, pd)
    prompt = _load_prompt("extraction_procedural.txt", _DEFAULT_PROCEDURAL_PROMPT, pd)
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
) -> SessionGraph:
    """Extract preferences/habits from a session transcript.

    Separate extraction pass with a dedicated prompt targeting
    behavioral patterns rather than factual knowledge.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_procedural_prompt(prompts_dir)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(transcript=transcript)},
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
    temperature: float = 0.3,
    max_tokens: int = 1024,
    prompts_dir: str | Path | None = None,
    validate: bool = True,
    ha_context: dict | None = None,
    stt_correction: bool = True,
    ha_validation: bool = True,
    noise_filter: str = "",
    noise_filter_model: str = "claude-sonnet-4-6",
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Multi-pass pipeline:
    1. Extract candidate triples from transcript
    2. Correct STT entity names from assistant responses (configurable)
    3. Validate with HA context — location ground truth (configurable)
    4. SOTA noise filter — anonymize → cloud filter → de-anonymize (configurable)

    All filters fail gracefully — extraction result is preserved on any failure.

    Args:
        prompts_dir: Optional override for prompt config directory.
        validate: Run SOTA noise filter pass 4 (default True). Passes 2-3 have
            their own flags (stt_correction, ha_validation).
        ha_context: HA home config for location validation (from get_home_context).
        stt_correction: Correct entity names from assistant responses.
        ha_validation: Validate locations against HA home context.
        noise_filter: SOTA provider for noise filtering ("" = disabled).
    """
    raw_output = _generate_extraction(
        model, tokenizer, transcript, temperature, max_tokens, prompts_dir
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

    # Pass 4: SOTA noise filter (anonymize → cloud filter → de-anonymize)
    if validate and noise_filter and graph.relations:
        graph = _sota_noise_filter(
            graph,
            transcript,
            model,
            tokenizer,
            provider=noise_filter,
            filter_model=noise_filter_model,
        )

    return graph


def _generate_extraction(
    model,
    tokenizer,
    transcript: str,
    temperature: float,
    max_tokens: int,
    prompts_dir: str | Path | None = None,
) -> str:
    """Generate graph extraction output from the model. Called once."""
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_extraction_prompts(prompts_dir)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(transcript=transcript)},
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


# Field name mappings for normalization
_ENTITY_TYPE_ALIASES = {"person", "place", "organization", "concept", "preference", "event"}
_RELATION_TYPE_ALIASES = {"factual", "temporal", "preference", "social"}


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
            norm["name"] = str(raw_name).strip().title()
            raw_type = ent.get("entity_type") or ent.get("type", "concept")
            if isinstance(raw_type, list):
                raw_type = raw_type[0] if raw_type else "concept"
            norm["entity_type"] = raw_type if raw_type in _ENTITY_TYPE_ALIASES else "concept"
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
            subject = str(raw_subj).strip().title()
            obj = str(raw_obj).strip().title()

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
            norm["relation_type"] = raw_type if raw_type in _RELATION_TYPE_ALIASES else "factual"
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
                clean = w.strip(".,!?;:'\"()").title()
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
Anonymize the following extracted personal facts by replacing all identifying \
information with category-prefixed placeholders.

Replace person names with Person_1, cities with City_1, etc.
Keep predicates, relation_type, and confidence unchanged.

Facts: {facts_json}

Return JSON: {{"anonymized": [...], "mapping": {{"Person_1": "name", ...}}}}
"""

_DEFAULT_NOISE_FILTER_PROMPT = """\
Review these extracted personal facts from a voice assistant conversation.

1. Resolve coreference ("my wife" → married_to relation).
2. Split compound facts into individual relations.
3. Remove noise: casual questions, device identifiers, implausible entities.

Conversation transcript (anonymized):
{transcript}

Extracted facts (anonymized):
{facts_json}

Return ONLY a JSON array of enriched facts. Each fact: subject, predicate, object, \
relation_type, confidence. If none survive, return [].
"""


def _sota_noise_filter(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
) -> SessionGraph:
    """Enrich and filter extraction via local anonymization → SOTA cloud validation.

    Three-in-one SOTA pass:
    1. Coreference resolution — "my wife" → married_to(speaker, wife_name)
    2. Compound fact splitting — "grew up in A and studied in B" → two relations
    3. Noise removal — device IDs, casual queries, implausible entities

    Pipeline: anonymize locally → SOTA enriches + filters → de-anonymize.
    Falls back to unfiltered graph on any failure.
    """
    import os

    # Determine API key based on provider
    key_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    api_key = os.environ.get(key_env.get(provider, "ANTHROPIC_API_KEY"), "")
    if not api_key:
        logger.info("No API key for %s — skipping SOTA enrichment", provider)
        return graph

    original_count = len(graph.relations)

    # Step 1: Anonymize facts and transcript via local model
    anon_facts, mapping = _anonymize_with_local_model(graph, model, tokenizer)
    if anon_facts is None:
        logger.warning("Anonymization failed — skipping SOTA enrichment")
        return graph

    # Apply the same mapping to the transcript for conversation context
    anon_transcript = _anonymize_transcript(transcript, mapping)

    # Step 2: SOTA enrichment — coreference + splitting + noise filtering
    enriched_anon = _filter_with_sota(anon_facts, api_key, provider, filter_model, anon_transcript)
    if enriched_anon is None:
        logger.warning("SOTA enrichment failed — keeping original")
        return graph

    if not enriched_anon:
        logger.info("SOTA enrichment removed all relations")
        graph.relations = []
        graph.entities = []
        return graph

    # Step 3: De-anonymize — reverse the mapping for all surviving/new relations
    reverse_mapping = {v: k for k, v in mapping.items()}
    from paramem.graph.schema import Entity, Relation

    kept_relations = []
    for fact in enriched_anon:
        if not isinstance(fact, dict):
            continue
        subj = reverse_mapping.get(fact.get("subject", ""), fact.get("subject", ""))
        obj = reverse_mapping.get(fact.get("object", ""), fact.get("object", ""))
        try:
            kept_relations.append(
                Relation(
                    subject=subj,
                    predicate=fact.get("predicate", ""),
                    object=obj,
                    relation_type=fact.get("relation_type", "factual"),
                    confidence=float(fact.get("confidence", 1.0)),
                )
            )
        except Exception:
            continue

    # Rebuild entity list from surviving + new relations
    kept_names = {r.subject for r in kept_relations} | {r.object for r in kept_relations}
    existing_names = {e.name for e in graph.entities}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    # Add entities for newly resolved coreferences
    for name in kept_names - existing_names:
        # Infer entity type from placeholder prefix (Person_, City_, Org_, etc.)
        entity_type = "person"
        placeholder = reverse_mapping.get(name)
        if placeholder:
            prefix = placeholder.split("_")[0].lower()
            type_map = {"person": "person", "city": "location", "org": "organization"}
            entity_type = type_map.get(prefix, "person")
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


def _anonymize_transcript(transcript: str, mapping: dict) -> str:
    """Apply entity name → placeholder mapping to a transcript.

    Replaces all mapped entity names with their anonymized placeholders
    so SOTA can see the conversation context without identifying info.
    Longer names are replaced first to avoid partial matches.
    """
    result = transcript
    for original, placeholder in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        result = result.replace(original, placeholder)
    return result


def _anonymize_with_local_model(
    graph: SessionGraph, model, tokenizer
) -> tuple[list[dict] | None, dict]:
    """Anonymize extracted facts using the local model.

    Returns (anonymized_facts, mapping) or (None, {}) on failure.
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

    anon_prompt = _load_prompt("anonymization.txt", _DEFAULT_ANONYMIZATION_PROMPT)
    prompt = anon_prompt.format(facts_json=json.dumps(facts, indent=2))
    messages = [
        {"role": "system", "content": "You anonymize data. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )

    raw = generate_answer(model, tokenizer, formatted, max_new_tokens=2048, temperature=0.0)
    logger.debug("Anonymization raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        if isinstance(data, dict) and "anonymized" in data and "mapping" in data:
            return data["anonymized"], data["mapping"]
        logger.warning("Anonymization returned unexpected format")
        return None, {}
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Anonymization parse failed: %s", e)
        return None, {}


def _filter_with_sota(
    anon_facts: list[dict],
    api_key: str,
    provider: str = "anthropic",
    filter_model: str = "claude-sonnet-4-6",
    anon_transcript: str | None = None,
) -> list[dict] | None:
    """Send anonymized facts + transcript to SOTA for enrichment and filtering."""
    if provider != "anthropic":
        logger.warning(
            "SOTA noise filter only supports 'anthropic' provider, got '%s' — "
            "set consolidation.extraction_noise_filter to 'anthropic' or '' to disable",
            provider,
        )
        return None

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping SOTA filter")
        return None

    filter_prompt = _load_prompt("noise_filter.txt", _DEFAULT_NOISE_FILTER_PROMPT)
    prompt = filter_prompt.format(
        facts_json=json.dumps(anon_facts, indent=2),
        transcript=anon_transcript or "(not available)",
    )

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
        response = client.messages.create(
            model=filter_model,
            max_tokens=2048,
            system="You are a knowledge graph enrichment assistant. Output valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        logger.warning("SOTA API call failed: %s", e)
        return None

    logger.debug("SOTA filter raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        validated = json.loads(json_str)
        if isinstance(validated, dict):
            for key in ("relations", "filtered", "facts", "results"):
                if key in validated and isinstance(validated[key], list):
                    return validated[key]
        if isinstance(validated, list):
            return validated
        logger.warning("SOTA filter returned unexpected format: %s", type(validated).__name__)
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("SOTA filter parse failed: %s", e)
        return None
