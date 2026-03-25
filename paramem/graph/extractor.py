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


def extract_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    prompts_dir: str | Path | None = None,
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Generates once, then tries structured parsing. Never regenerates —
    the first generation at temp=0 produces the best output.

    Args:
        prompts_dir: Optional override for prompt config directory.
    """
    raw_output = _generate_extraction(
        model, tokenizer, transcript, temperature, max_tokens, prompts_dir
    )
    logger.debug("Raw extraction output: %s", raw_output[:500])

    try:
        return _parse_extraction(raw_output, session_id)
    except Exception as exc:
        logger.warning(
            "Extraction parsing failed (%s), returning empty graph",
            exc,
        )
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


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
