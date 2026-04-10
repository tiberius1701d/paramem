#!/usr/bin/env python3
"""Compare extraction quality: Mistral vs Claude — same pipeline, validation ON.

Both models go through the same code path:
1. Same extraction prompt
2. Same JSON parsing
3. Same normalization
4. Same STT correction
5. Same validation pass

Usage:
    export $(grep -v '^#' .env | xargs)
    python scripts/compare_extraction.py
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paramem.graph.extractor import (
    _correct_entity_names,
    _extract_json_block,
    _normalize_extraction,
    load_extraction_prompts,
)
from paramem.graph.schema import SessionGraph

# Enable DEBUG logging to see raw validation output
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
# Quiet noisy loggers
for name in ("httpx", "anthropic", "urllib3", "transformers", "accelerate", "bitsandbytes"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path("configs/prompts")
SESSION_DIR = Path("data/ha/sessions")
OUTPUT_DIR = Path("data/ha/debug")


def load_transcripts(session_dir: Path) -> list[dict]:
    """Load all transcripts from session JSONL files."""
    transcripts = []
    for f in sorted(session_dir.glob("*.jsonl")):
        lines = f.read_text().strip().split("\n")
        turns = [json.loads(line) for line in lines]
        text_parts = []
        for turn in turns:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            text_parts.append(f"[{role}] {text}")
        transcript = "\n".join(text_parts)
        if len(transcript) > 50:
            transcripts.append({"session_id": f.stem, "transcript": transcript})
    return transcripts


def run_extraction(
    raw_output: str, transcript: str, session_id: str, model=None, tokenizer=None, validate=True
) -> SessionGraph:
    """Shared extraction pipeline: parse → normalize → STT correct → validate."""
    try:
        json_str = _extract_json_block(raw_output)
        data = json.loads(json_str)
        data["session_id"] = session_id
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data = _normalize_extraction(data)
        graph = SessionGraph.model_validate(data)
    except Exception as e:
        logger.warning("Parse error: %s", e)
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if not graph.relations:
        return graph

    # STT correction
    graph = _correct_entity_names(graph, transcript)

    # Validation is now handled by the SOTA noise filter in extract_graph().
    # This comparison script only runs extraction + STT correction.

    return graph


def generate_with_claude(transcript: str) -> str:
    """Get raw extraction output from Claude."""
    import anthropic

    system, prompt = load_extraction_prompts(PROMPT_DIR)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt.format(transcript=transcript)}],
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def generate_with_mistral(transcript: str, model, tokenizer) -> str:
    """Get raw extraction output from Mistral."""
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_extraction_prompts(PROMPT_DIR)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(transcript=transcript)},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate_answer(model, tokenizer, formatted, max_new_tokens=2048, temperature=0.0)


def validate_with_claude(graph: SessionGraph, transcript: str) -> SessionGraph:
    """Run the SOTA noise filter prompt through Claude."""
    import anthropic

    from paramem.graph.extractor import _extract_json_block, _load_prompt
    from paramem.graph.schema import Relation

    noise_prompt = _load_prompt("noise_filter.txt", "")
    if not noise_prompt:
        logger.warning("No noise_filter.txt found, skipping validation")
        return graph

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
    excerpt = transcript[:1000]
    prompt = noise_prompt.format(
        facts_json=json.dumps(facts, indent=2),
        transcript=excerpt,
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a fact validation assistant. Output valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    logger.debug("Claude validation raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        validated = json.loads(json_str)
        if isinstance(validated, dict):
            for key in ("relations", "filtered", "facts", "results"):
                if key in validated and isinstance(validated[key], list):
                    validated = validated[key]
                    break
        if not isinstance(validated, list):
            logger.warning("Claude validation returned non-list, keeping original")
            return graph
    except (json.JSONDecodeError, ValueError):
        logger.warning("Claude validation parse failed, keeping original")
        return graph

    if not validated:
        graph.relations = []
        graph.entities = []
        return graph

    kept = []
    for v in validated:
        if not isinstance(v, dict):
            continue
        try:
            kept.append(
                Relation(
                    subject=v.get("subject", ""),
                    predicate=v.get("predicate", ""),
                    object=v.get("object", ""),
                    relation_type=v.get("relation_type", "factual"),
                    confidence=float(v.get("confidence", 1.0)),
                )
            )
        except Exception:
            continue

    kept_names = set()
    for r in kept:
        kept_names.add(r.subject)
        kept_names.add(r.object)
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    graph.relations = kept
    logger.info("Claude validation: %d/%d relations kept", len(kept), len(facts))
    return graph


def print_relations(label: str, graph: SessionGraph):
    """Print relations for one model."""
    if graph.relations:
        for r in graph.relations:
            print(
                f"    {r.subject} --[{r.predicate}]--> {r.object}"
                f"  ({r.relation_type}, conf={r.confidence})"
            )
    else:
        print("    (none)")


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    transcripts = load_transcripts(SESSION_DIR)
    if not transcripts:
        transcripts = load_transcripts(SESSION_DIR / "archive")
    print(f"Processing {len(transcripts)} sessions\n")
    if not transcripts:
        print("No sessions found")
        sys.exit(0)

    # Load Mistral
    print("Loading Mistral model...")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.utils.config import load_config

    config = load_config()
    model, tokenizer = load_base_model(config.model)
    print("Model loaded\n")

    results = []
    for i, t in enumerate(transcripts):
        sid = t["session_id"][:20]
        preview = t["transcript"].split("\n")[0][:80]
        print(f"{'=' * 70}")
        print(f"Session {i + 1}/{len(transcripts)}: {sid}")
        print(f"  {preview}")
        print()

        # --- Claude: extract → normalize → STT correct → validate (via Claude) ---
        print("  CLAUDE (extraction):")
        claude_raw = generate_with_claude(t["transcript"])
        claude_graph_raw = run_extraction(
            claude_raw,
            t["transcript"],
            t["session_id"],
            model=None,
            tokenizer=None,
            validate=False,
        )
        print_relations("Claude (pre-validation)", claude_graph_raw)

        print("  CLAUDE (after validation):")
        claude_graph = run_extraction(
            claude_raw,
            t["transcript"],
            t["session_id"],
            model=None,
            tokenizer=None,
            validate=False,
        )
        if claude_graph.relations:
            claude_graph = validate_with_claude(claude_graph, t["transcript"])
        print_relations("Claude (validated)", claude_graph)

        # --- Mistral: extract → normalize → STT correct → validate (via Mistral) ---
        print("  MISTRAL (extraction):")
        mistral_raw = generate_with_mistral(t["transcript"], model, tokenizer)
        mistral_graph_raw = run_extraction(
            mistral_raw,
            t["transcript"],
            t["session_id"],
            model=None,
            tokenizer=None,
            validate=False,
        )
        print_relations("Mistral (pre-validation)", mistral_graph_raw)

        print("  MISTRAL (after validation):")
        mistral_graph = run_extraction(
            mistral_raw,
            t["transcript"],
            t["session_id"],
            model=model,
            tokenizer=tokenizer,
            validate=True,
        )
        print_relations("Mistral (validated)", mistral_graph)

        results.append(
            {
                "session_id": t["session_id"],
                "transcript_preview": t["transcript"][:300],
                "claude_raw": [
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "relation_type": r.relation_type,
                    }
                    for r in claude_graph_raw.relations
                ],
                "claude_validated": [
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "relation_type": r.relation_type,
                    }
                    for r in claude_graph.relations
                ],
                "mistral_raw": [
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "relation_type": r.relation_type,
                    }
                    for r in mistral_graph_raw.relations
                ],
                "mistral_validated": [
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "relation_type": r.relation_type,
                    }
                    for r in mistral_graph.relations
                ],
            }
        )
        print()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "extraction_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("=" * 75)
    print("SUMMARY")
    print(f"{'Session':<25} {'C_raw':>6} {'C_val':>6} {'M_raw':>6} {'M_val':>6}")
    print("-" * 75)
    for r in results:
        sid = r["session_id"][:22]
        cr = len(r["claude_raw"])
        cv = len(r["claude_validated"])
        mr = len(r["mistral_raw"])
        mv = len(r["mistral_validated"])
        print(f"  {sid:<25} {cr:>4}   {cv:>4}   {mr:>4}   {mv:>4}")
    tcr = sum(len(r["claude_raw"]) for r in results)
    tcv = sum(len(r["claude_validated"]) for r in results)
    tmr = sum(len(r["mistral_raw"]) for r in results)
    tmv = sum(len(r["mistral_validated"]) for r in results)
    print("-" * 75)
    print(f"  {'TOTAL':<25} {tcr:>4}   {tcv:>4}   {tmr:>4}   {tmv:>4}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
