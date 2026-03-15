"""LLM-based knowledge graph extraction using Outlines constrained generation."""

import json
import logging
from datetime import datetime, timezone

from paramem.graph.schema import SessionGraph

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = "You are a precise knowledge graph extractor. Output valid JSON only."

EXTRACTION_PROMPT = """\
Extract all entities and relations from this conversation transcript.

Rules:
- Entity types: person, place, organization, concept, preference, event
- Relation types: factual, temporal, preference, social
- Use canonical names (e.g. "Alex" not "the user")
- Include attributes like age, role, date where mentioned
- predicate: snake_case verb phrase — consistent, reusable across sessions
- confidence: 1.0 for explicit facts, 0.7 for implied/inferred

Predicate examples (use these exact forms when they apply):
- lives_in, works_at, studies_at, prefers, knows, has_pet, has_hobby
- manages, attended, visited, bought, read, fixed, cooked, watched
- If none fit, create a new snake_case predicate (e.g. "competed_in")

Example output:
{{"entities": [{{"name": "Alex", "entity_type": "person", "attributes": {{}}}},\
 {{"name": "Heilbronn", "entity_type": "place", "attributes": {{}}}}],\
 "relations": [{{"subject": "Alex", "predicate": "lives_in",\
 "object": "Heilbronn", "relation_type": "factual", "confidence": 1.0}}],\
 "summary": "Alex lives in Heilbronn."}}

Transcript:
{transcript}

Extract all entities and relations as JSON."""


def extract_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Uses Outlines constrained generation to guarantee valid JSON output
    matching the SessionGraph schema.
    """
    try:
        return _extract_with_outlines(
            model, tokenizer, transcript, session_id, temperature, max_tokens
        )
    except Exception as exc:
        logger.warning(
            "Outlines extraction failed (%s), falling back to prompt-and-parse",
            exc,
        )
    try:
        return _extract_with_prompt_parse(
            model, tokenizer, transcript, session_id, temperature, max_tokens
        )
    except Exception as exc:
        logger.warning(
            "Prompt-parse extraction also failed (%s), returning empty graph",
            exc,
        )
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _extract_with_outlines(
    model, tokenizer, transcript, session_id, temperature, max_tokens
) -> SessionGraph:
    """Extract using Outlines constrained JSON generation."""
    import outlines

    from paramem.models.loader import adapt_messages

    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer), tokenize=False, add_generation_prompt=True
    )

    outlines_model = outlines.from_transformers(model, tokenizer)
    generator = outlines.Generator(outlines_model, SessionGraph)

    result = generator(formatted_prompt, max_tokens=max_tokens)

    # Outlines returns a SessionGraph directly when given the schema
    if isinstance(result, SessionGraph):
        graph = result
    else:
        graph = SessionGraph.model_validate(result)

    # Override session metadata
    graph.session_id = session_id
    graph.timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Extracted graph: %d entities, %d relations (session=%s)",
        len(graph.entities),
        len(graph.relations),
        session_id,
    )
    return graph


def _extract_with_prompt_parse(
    model, tokenizer, transcript, session_id, temperature, max_tokens
) -> SessionGraph:
    """Fallback: extract using free-form generation + JSON parsing."""
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript)},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer), tokenize=False, add_generation_prompt=True
    )

    raw_output = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.1),
    )

    # Try to find JSON in the output
    json_str = _extract_json_block(raw_output)
    data = json.loads(json_str)

    data["session_id"] = session_id
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Normalize field names — model may use different keys
    data = _normalize_extraction(data)

    graph = SessionGraph.model_validate(data)
    logger.info(
        "Extracted graph (fallback): %d entities, %d relations (session=%s)",
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
            end = text.index("```", start)
            return text[start:end].strip()

    # Fall back to finding raw JSON object by brace matching
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError("No JSON found in model output")

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]

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
            norm = {}
            raw_name = ent.get("name") or ent.get("entity", "unknown")
            norm["name"] = raw_name.strip().title()
            raw_type = ent.get("entity_type") or ent.get("type", "concept")
            norm["entity_type"] = raw_type if raw_type in _ENTITY_TYPE_ALIASES else "concept"
            norm["attributes"] = ent.get("attributes", {})
            # If model put extra fields as top-level, capture them as strings
            skip_keys = {"name", "entity", "entity_type", "type", "attributes"}
            for k, v in ent.items():
                if k not in skip_keys and v is not None:
                    norm["attributes"][k] = str(v)
            normalized_entities.append(norm)
        data["entities"] = normalized_entities

    # Normalize relations
    if "relations" in data:
        normalized_relations = []
        for rel in data["relations"]:
            subject = rel.get("subject", "unknown").strip().title()
            obj = rel.get("object", "unknown").strip().title()

            # Filter self-loops (e.g. "KIT studied at KIT")
            if subject.lower() == obj.lower():
                logger.debug("Filtered self-loop: %s -> %s", subject, obj)
                continue

            norm = {
                "subject": subject,
                "predicate": rel.get("predicate", "related_to"),
                "object": obj,
                "confidence": rel.get("confidence", 1.0),
            }
            raw_type = rel.get("relation_type") or rel.get("type", "factual")
            norm["relation_type"] = raw_type if raw_type in _RELATION_TYPE_ALIASES else "factual"
            normalized_relations.append(norm)
        data["relations"] = normalized_relations

    # Ensure required top-level fields
    data.setdefault("summary", "")
    data.setdefault("entities", [])
    data.setdefault("relations", [])

    return data
