#!/usr/bin/env python3
"""Compare extraction quality: Claude vs Mistral vs Gemma 4 — same pipeline.

All models go through the same code path:
1. Same extraction prompt
2. Same JSON parsing
3. Same normalization
4. Same STT correction

Each model pass saves results immediately. Re-running skips completed passes.

Usage:
    # Stop the server first (frees GPU VRAM)
    systemctl --user stop paramem-server

    export $(grep -v '^#' .env | xargs)
    python scripts/compare_extraction.py

    # Re-run only missing passes (e.g. after a crash):
    python scripts/compare_extraction.py

    # Force re-run a specific model:
    python scripts/compare_extraction.py --rerun gemma4
"""

import argparse
import gc
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

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
for name in ("httpx", "anthropic", "urllib3", "transformers", "accelerate", "bitsandbytes"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path("configs/prompts")
SESSION_DIR = Path("data/ha/sessions")
OUTPUT_DIR = Path("data/ha/debug/extraction_comparison")


def load_transcripts() -> list[dict]:
    """Load all transcripts from session JSONL files (pending + archived)."""
    transcripts = []
    for search_dir in [SESSION_DIR, SESSION_DIR / "archive"]:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.glob("*.jsonl")):
            if f.stem == "test-lang":
                continue
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


def run_extraction(raw_output: str, transcript: str, session_id: str) -> SessionGraph:
    """Shared extraction pipeline: parse -> normalize -> STT correct."""
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

    graph = _correct_entity_names(graph, transcript)
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


def generate_with_local(transcript: str, model, tokenizer) -> str:
    """Get raw extraction output from a local model."""
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

    try:
        json_str = _extract_json_block(raw)
        validated = json.loads(json_str)
        if isinstance(validated, dict):
            for key in ("relations", "filtered", "facts", "results"):
                if key in validated and isinstance(validated[key], list):
                    validated = validated[key]
                    break
        if not isinstance(validated, list):
            return graph
    except (json.JSONDecodeError, ValueError):
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


def serialize_relations(graph: SessionGraph) -> list[dict]:
    return [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
        }
        for r in graph.relations
    ]


def print_relations(graph: SessionGraph):
    if graph.relations:
        for r in graph.relations:
            print(
                f"    {r.subject} --[{r.predicate}]--> {r.object}"
                f"  ({r.relation_type}, conf={r.confidence})"
            )
    else:
        print("    (none)")


def save_pass(model_name: str, results: dict):
    """Save a single model pass to disk immediately."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [{model_name}] Results saved to {path}")


def load_pass(model_name: str) -> dict | None:
    """Load a previously completed model pass."""
    path = OUTPUT_DIR / f"{model_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def unload_model(model, tokenizer):
    """Free GPU memory after a model pass."""
    import torch

    if hasattr(model, "cpu"):
        model.cpu()
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# --- Model passes ---


def run_claude_pass(transcripts: list[dict]) -> dict:
    """Run Claude extraction on all sessions."""
    results = {}
    for i, t in enumerate(transcripts):
        sid = t["session_id"]
        preview = t["transcript"].split("\n")[0][:80]
        print(f"  [claude] Session {i + 1}/{len(transcripts)}: {sid[:20]}")
        print(f"    {preview}")

        raw = generate_with_claude(t["transcript"])
        graph = run_extraction(raw, t["transcript"], sid)
        print(f"    -> {len(graph.relations)} relations (raw)")
        print_relations(graph)

        validated = graph
        if graph.relations:
            validated = validate_with_claude(graph, t["transcript"])
            print(f"    -> {len(validated.relations)} relations (validated)")

        results[sid] = {
            "raw_output": raw,
            "relations_raw": serialize_relations(graph),
            "relations_validated": serialize_relations(validated),
        }

    return results


def run_local_pass(model_name: str, transcripts: list[dict]) -> dict:
    """Load a local model, extract all sessions, save, unload."""
    from paramem.models.loader import load_base_model
    from paramem.server.config import MODEL_REGISTRY

    model_config = MODEL_REGISTRY[model_name]
    print(f"  Loading {model_name}...")
    model, tokenizer = load_base_model(model_config)
    print(f"  {model_name} loaded\n")

    results = {}
    for i, t in enumerate(transcripts):
        sid = t["session_id"]
        preview = t["transcript"].split("\n")[0][:80]
        print(f"  [{model_name}] Session {i + 1}/{len(transcripts)}: {sid[:20]}")
        print(f"    {preview}")

        raw = generate_with_local(t["transcript"], model, tokenizer)
        graph = run_extraction(raw, t["transcript"], sid)
        print(f"    -> {len(graph.relations)} relations")
        print_relations(graph)

        results[sid] = {
            "raw_output": raw,
            "relations": serialize_relations(graph),
        }

    print(f"\n  Unloading {model_name}...")
    unload_model(model, tokenizer)
    print(f"  {model_name} unloaded")

    return results


def print_summary(transcripts, claude, mistral, gemma4):
    """Print comparison summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"{'Session':<25} {'C_raw':>6} {'C_val':>6} {'M_raw':>6} {'G4_raw':>7}")
    print("-" * 80)
    for t in transcripts:
        sid = t["session_id"][:22]
        cr = len(claude.get(t["session_id"], {}).get("relations_raw", []))
        cv = len(claude.get(t["session_id"], {}).get("relations_validated", []))
        mr = len(mistral.get(t["session_id"], {}).get("relations", []))
        gr = len(gemma4.get(t["session_id"], {}).get("relations", []))
        print(f"  {sid:<25} {cr:>4}   {cv:>4}   {mr:>4}   {gr:>5}")
    tcr = sum(len(v.get("relations_raw", [])) for v in claude.values())
    tcv = sum(len(v.get("relations_validated", [])) for v in claude.values())
    tmr = sum(len(v.get("relations", [])) for v in mistral.values())
    tgr = sum(len(v.get("relations", [])) for v in gemma4.values())
    print("-" * 80)
    print(f"  {'TOTAL':<25} {tcr:>4}   {tcv:>4}   {tmr:>4}   {tgr:>5}")


def main():
    parser = argparse.ArgumentParser(description="Compare extraction quality across models")
    parser.add_argument("--rerun", help="Force re-run a specific model pass", default=None)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    transcripts = load_transcripts()
    print(f"Processing {len(transcripts)} sessions\n")
    if not transcripts:
        print("No sessions found")
        sys.exit(0)

    # --- Pass 1: Claude (API) ---
    claude = load_pass("claude") if args.rerun != "claude" else None
    if claude is not None:
        print("PASS 1: Claude — loaded from previous run")
    else:
        print("=" * 70)
        print("PASS 1: Claude (API)")
        print("=" * 70)
        claude = run_claude_pass(transcripts)
        save_pass("claude", claude)

    # --- Pass 2: Mistral (GPU) ---
    mistral = load_pass("mistral") if args.rerun != "mistral" else None
    if mistral is not None:
        print("PASS 2: Mistral — loaded from previous run")
    else:
        print("\n" + "=" * 70)
        print("PASS 2: Mistral (GPU)")
        print("=" * 70)
        mistral = run_local_pass("mistral", transcripts)
        save_pass("mistral", mistral)

    # --- Pass 3: Gemma 4 (GPU) ---
    gemma4 = load_pass("gemma4") if args.rerun != "gemma4" else None
    if gemma4 is not None:
        print("PASS 3: Gemma 4 — loaded from previous run")
    else:
        print("\n" + "=" * 70)
        print("PASS 3: Gemma 4 (GPU)")
        print("=" * 70)
        gemma4 = run_local_pass("gemma4", transcripts)
        save_pass("gemma4", gemma4)

    print_summary(transcripts, claude, mistral, gemma4)


if __name__ == "__main__":
    main()
