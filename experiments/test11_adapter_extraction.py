"""Test 11: Adapter-Active Extraction — Quality Comparison.

Compares graph extraction quality with the LoRA adapter ON vs OFF.
Uses the Test 8 cycle 50 adapter (528 keys) and PerLTQA transcripts.

Question: should the pipeline extract with a pretrained adapter active,
or with the clean base model for reproducible extraction?

Design:
  - Two fully isolated passes (fresh model load per condition)
  - max_tokens=2048 to eliminate format compliance as a confound
  - Raw model output saved for every extraction
  - Entity grounding: % of extracted entities found in transcript text
  - Triple grounding: % of triples where both subject and object appear

Usage:
    python experiments/test11_adapter_extraction.py --model mistral
    python experiments/test11_adapter_extraction.py --model mistral --num-sessions 10
    python experiments/test11_adapter_extraction.py --model mistral --resume
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PAUSE_FILE = Path.home() / ".training_pause"
MAX_EXTRACTION_TOKENS = 2048


def is_paused():
    """Check if pause has been requested via tpause."""
    return PAUSE_FILE.exists()


def wait_for_cooldown(target=45):
    """Block until GPU temperature drops below target."""
    try:
        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}",
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Cooldown script failed (%s), falling back to 60s sleep", e)
        time.sleep(60)


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from experiments.utils.perltqa_loader import (  # noqa: E402
    list_characters,
    load_character_dialogues,
)
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    model_output_dir,
    setup_logging,
)
from paramem.graph.extractor import (  # noqa: E402
    _generate_extraction,
    _parse_extraction,
    load_extraction_prompts,
)
from paramem.models.loader import (  # noqa: E402
    load_adapter,
    load_base_model,
)

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "outputs" / "test11_adapter_extraction"
TEST8_RUN_DIR = project_root / "outputs" / "test8_large_scale" / "mistral" / "20260323_161747"
ADAPTER_NAME = "episodic"


def save_json_atomic(data, target: Path):
    """Write JSON atomically via temp file + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(tmp_path).replace(target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ============================================================================
# Session selection
# ============================================================================


def select_sessions(num_sessions: int = 50) -> list[dict]:
    """Select sessions from PerLTQA, prioritizing characters with most dialogues.

    Same priority order as Test 8: characters sorted by dialogue count descending.
    """
    characters = list_characters()
    sorted_chars = sorted(characters.items(), key=lambda x: -x[1]["dialogues"])

    sessions = []
    for char_name, info in sorted_chars:
        if len(sessions) >= num_sessions:
            break
        char_sessions = load_character_dialogues(char_name)
        for s in char_sessions:
            if len(sessions) >= num_sessions:
                break
            s["character"] = char_name
            sessions.append(s)

    logger.info(
        "Selected %d sessions from %d characters",
        len(sessions),
        len({s["character"] for s in sessions}),
    )
    return sessions


# ============================================================================
# Extraction with raw output capture
# ============================================================================


def extract_session(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
) -> dict:
    """Extract graph from a single transcript, capturing raw output.

    Model must be loaded in the correct configuration by the caller.
    Returns structured result dict with raw output for diagnostics.
    """
    # Generate raw output
    try:
        raw_output = _generate_extraction(
            model, tokenizer, transcript, temperature=0.0, max_tokens=MAX_EXTRACTION_TOKENS
        )
    except Exception as e:
        logger.warning("Generation failed for %s: %s", session_id, e)
        return _empty_result(str(e), raw_output="")

    output_tokens = len(tokenizer.encode(raw_output))
    truncated = output_tokens >= MAX_EXTRACTION_TOKENS

    # Parse
    try:
        sg = _parse_extraction(raw_output, session_id)
    except Exception as e:
        logger.warning("Parse failed for %s: %s", session_id, e)
        return _empty_result(
            str(e), raw_output=raw_output, output_tokens=output_tokens, truncated=truncated
        )

    relations = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "confidence": getattr(r, "confidence", None),
        }
        for r in sg.relations
    ]
    entities = [
        {
            "name": e.name,
            "entity_type": getattr(e, "entity_type", None),
        }
        for e in sg.entities
    ]

    return {
        "success": len(relations) > 0,
        "triple_count": len(relations),
        "entity_count": len(entities),
        "unique_predicates": list({r["predicate"] for r in relations}),
        "predicate_count": len({r["predicate"] for r in relations}),
        "relations": relations,
        "entities": entities,
        "raw_output": raw_output,
        "output_tokens": output_tokens,
        "truncated": truncated,
        "error": None,
    }


def _empty_result(
    error: str, raw_output: str = "", output_tokens: int = 0, truncated: bool = False
) -> dict:
    return {
        "success": False,
        "triple_count": 0,
        "entity_count": 0,
        "unique_predicates": [],
        "predicate_count": 0,
        "relations": [],
        "entities": [],
        "raw_output": raw_output,
        "output_tokens": output_tokens,
        "truncated": truncated,
        "error": error,
    }


# ============================================================================
# Grounding metrics — is the extraction faithful to the transcript?
# ============================================================================


def compute_entity_grounding(result: dict, transcript: str) -> dict:
    """Check what fraction of extracted entities appear in the transcript.

    An entity is "grounded" if its name (case-insensitive) appears as a
    substring in the transcript text. Ungrounded entities may be hallucinated
    from adapter training data.
    """
    transcript_lower = transcript.lower()
    entities = result.get("entities", [])
    if not entities:
        return {"grounded": 0, "total": 0, "rate": 0.0, "ungrounded": []}

    grounded = 0
    ungrounded = []
    for ent in entities:
        name = ent["name"].lower()
        if name in transcript_lower:
            grounded += 1
        else:
            ungrounded.append(ent["name"])

    return {
        "grounded": grounded,
        "total": len(entities),
        "rate": round(grounded / len(entities), 4),
        "ungrounded": ungrounded,
    }


def compute_triple_grounding(result: dict, transcript: str) -> dict:
    """Check what fraction of triples have both subject and object in transcript.

    A triple is "grounded" if both its subject and object names appear in the
    transcript. A triple with an ungrounded subject or object may reflect
    adapter training priors rather than transcript content.
    """
    transcript_lower = transcript.lower()
    relations = result.get("relations", [])
    if not relations:
        return {"grounded": 0, "total": 0, "rate": 0.0, "ungrounded_triples": []}

    grounded = 0
    ungrounded_triples = []
    for rel in relations:
        subj_in = rel["subject"].lower() in transcript_lower
        obj_in = rel["object"].lower() in transcript_lower
        if subj_in and obj_in:
            grounded += 1
        else:
            ungrounded_triples.append(
                {
                    "subject": rel["subject"],
                    "predicate": rel["predicate"],
                    "object": rel["object"],
                    "subject_grounded": subj_in,
                    "object_grounded": obj_in,
                }
            )

    return {
        "grounded": grounded,
        "total": len(relations),
        "rate": round(grounded / len(relations), 4),
        "ungrounded_triples": ungrounded_triples,
    }


# ============================================================================
# Comparison and aggregation
# ============================================================================


def compute_session_comparison(result_a: dict, result_b: dict) -> dict:
    """Compute per-session comparison metrics between conditions A and B."""
    preds_a = set(result_a.get("unique_predicates", []))
    preds_b = set(result_b.get("unique_predicates", []))
    union = preds_a | preds_b
    intersection = preds_a & preds_b

    # Triple overlap by (subject, predicate, object) identity
    triples_a = {
        (r["subject"].lower(), r["predicate"].lower(), r["object"].lower())
        for r in result_a.get("relations", [])
    }
    triples_b = {
        (r["subject"].lower(), r["predicate"].lower(), r["object"].lower())
        for r in result_b.get("relations", [])
    }
    triple_union = triples_a | triples_b
    triple_intersection = triples_a & triples_b

    return {
        "triple_diff": result_b["triple_count"] - result_a["triple_count"],
        "entity_diff": result_b["entity_count"] - result_a["entity_count"],
        "predicate_overlap": round(len(intersection) / len(union), 4) if union else 1.0,
        "triple_overlap": round(len(triple_intersection) / len(triple_union), 4)
        if triple_union
        else 1.0,
        "shared_triples": len(triple_intersection),
        "a_only_triples": len(triples_a - triples_b),
        "b_only_triples": len(triples_b - triples_a),
        "both_success": result_a["success"] and result_b["success"],
    }


def aggregate_results(session_results: list[dict]) -> dict:
    """Aggregate metrics across all sessions."""
    total = len(session_results)
    if total == 0:
        return {}

    success_a = sum(1 for sr in session_results if sr["adapter_off"]["success"])
    success_b = sum(1 for sr in session_results if sr["adapter_on"]["success"])
    both_success = sum(1 for sr in session_results if sr["comparison"]["both_success"])

    # Truncation stats
    trunc_a = sum(1 for sr in session_results if sr["adapter_off"].get("truncated", False))
    trunc_b = sum(1 for sr in session_results if sr["adapter_on"].get("truncated", False))

    def _mean(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    triples_a = [sr["adapter_off"]["triple_count"] for sr in session_results]
    triples_b = [sr["adapter_on"]["triple_count"] for sr in session_results]

    # Grounding (only for sessions with successful extraction)
    grounding_a = [
        sr["grounding_off"]["entity"]["rate"]
        for sr in session_results
        if sr["adapter_off"]["success"]
    ]
    grounding_b = [
        sr["grounding_on"]["entity"]["rate"]
        for sr in session_results
        if sr["adapter_on"]["success"]
    ]
    triple_grounding_a = [
        sr["grounding_off"]["triple"]["rate"]
        for sr in session_results
        if sr["adapter_off"]["success"]
    ]
    triple_grounding_b = [
        sr["grounding_on"]["triple"]["rate"]
        for sr in session_results
        if sr["adapter_on"]["success"]
    ]

    # Comparison metrics (only for sessions where both succeed)
    both_ok = [sr for sr in session_results if sr["comparison"]["both_success"]]

    return {
        "total_sessions": total,
        "max_tokens": MAX_EXTRACTION_TOKENS,
        "adapter_off": {
            "success_rate": round(success_a / total, 4),
            "truncated": trunc_a,
            "mean_triples": _mean(triples_a),
            "total_triples": sum(triples_a),
            "mean_entity_grounding": _mean(grounding_a),
            "mean_triple_grounding": _mean(triple_grounding_a),
        },
        "adapter_on": {
            "success_rate": round(success_b / total, 4),
            "truncated": trunc_b,
            "mean_triples": _mean(triples_b),
            "total_triples": sum(triples_b),
            "mean_entity_grounding": _mean(grounding_b),
            "mean_triple_grounding": _mean(triple_grounding_b),
        },
        "comparison": {
            "both_success_count": both_success,
            "mean_triple_overlap": _mean([sr["comparison"]["triple_overlap"] for sr in both_ok]),
            "mean_predicate_overlap": _mean(
                [sr["comparison"]["predicate_overlap"] for sr in both_ok]
            ),
            "mean_shared_triples": _mean([sr["comparison"]["shared_triples"] for sr in both_ok]),
        },
    }


# ============================================================================
# Main
# ============================================================================


def _free_cuda():
    """Force CUDA memory cleanup."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_pass(
    sessions: list[dict],
    model_name: str,
    adapter_path: Path | None,
    results_dir: Path,
    label: str,
    state: dict | None = None,
    state_path: Path | None = None,
) -> list[dict]:
    """Run extraction on all sessions with a freshly loaded model.

    Loads the model from scratch, runs all sessions, then fully unloads.
    If adapter_path is provided, loads the adapter onto the model.
    Saves per-session results to results_dir/{label}_session_NNN.json.
    Updates state file after each session if state/state_path provided.
    """
    model_config = BENCHMARK_MODELS[model_name]
    logger.info("=== Pass: %s — loading model from scratch ===", label)
    model, tokenizer = load_base_model(model_config)
    model.gradient_checkpointing_disable()

    if adapter_path is not None:
        logger.info("Loading adapter from %s", adapter_path)
        model = load_adapter(model, str(adapter_path), ADAPTER_NAME)

    results = []
    for i, session in enumerate(sessions):
        if is_paused():
            logger.info("Paused — stopping %s pass after %d sessions", label, i)
            break

        result_file = results_dir / f"{label}_session_{i + 1:03d}.json"
        if result_file.exists():
            logger.info(
                "%s %d/%d: %s — skipping (exists)",
                label,
                i + 1,
                len(sessions),
                session["session_id"],
            )
            with open(result_file) as f:
                results.append(json.load(f))
            continue

        logger.info(
            "%s %d/%d: %s (%s)",
            label,
            i + 1,
            len(sessions),
            session["session_id"],
            session["character"],
        )

        result = extract_session(
            model,
            tokenizer,
            session["transcript"],
            session["session_id"],
        )
        save_json_atomic(result, result_file)
        results.append(result)

        status = "OK" if result["success"] else "FAIL"
        logger.info(
            "  %s: %s — %d triples, %d entities, %d tokens%s",
            label,
            status,
            result["triple_count"],
            result["entity_count"],
            result.get("output_tokens", 0),
            " (TRUNCATED)" if result.get("truncated") else "",
        )

        # Update state for tstatus
        if state is not None and state_path is not None:
            state["completed_sessions"] = len(results)
            save_json_atomic(state, state_path)

        # Cooldown every 10 sessions
        if (i + 1) % 10 == 0 and i < len(sessions) - 1:
            wait_for_cooldown(52)

    logger.info("=== Pass: %s complete — %d sessions, unloading model ===", label, len(results))
    del model, tokenizer
    _free_cuda()
    return results


def run_experiment(model_name: str, num_sessions: int, resume: bool = False):
    """Main orchestration — two fully isolated passes.

    Pass 1: Load clean base model, extract all sessions (adapter OFF).
    Unload. Cooldown.
    Pass 2: Load base model + adapter, extract all sessions (adapter ON).
    Unload. Compute grounding and comparison metrics.
    """
    if model_name != "mistral":
        logger.error("Test 11 requires mistral (Test 8 adapter is Mistral-specific)")
        return

    adapter_path = TEST8_RUN_DIR / "cycle_050" / "adapter"
    if not adapter_path.exists():
        logger.error("Test 8 cycle 50 adapter not found: %s", adapter_path)
        return

    # Output directory
    if resume:
        model_dir = OUTPUT_BASE / model_name
        if model_dir.exists():
            subdirs = sorted(
                [d for d in model_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )
            if subdirs:
                output_dir = subdirs[0]
                logger.info("Resuming from: %s", output_dir)
            else:
                output_dir = model_output_dir(OUTPUT_BASE, model_name)
        else:
            output_dir = model_output_dir(OUTPUT_BASE, model_name)
    else:
        output_dir = model_output_dir(OUTPUT_BASE, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    if free_gb < 20:
        logger.error("Insufficient disk space: %.1f GB free", free_gb)
        return
    logger.info("Disk: %.1f GB free", free_gb)

    sessions = select_sessions(num_sessions)
    if not sessions:
        logger.error("No sessions available")
        return

    # Snapshot extraction prompts for reproducibility
    system_prompt, extraction_prompt = load_extraction_prompts()

    run_config = {
        "model": model_name,
        "num_sessions": len(sessions),
        "adapter_path": str(adapter_path),
        "characters": sorted({s["character"] for s in sessions}),
        "design": "two isolated passes — fresh model load per condition",
        "max_tokens": MAX_EXTRACTION_TOKENS,
        "extraction_system_prompt": system_prompt,
        "extraction_prompt_template": extraction_prompt,
    }
    save_json_atomic(run_config, output_dir / "run_config.json")

    state = {
        "total_sessions": len(sessions),
        "completed_sessions": 0,
        "model": model_name,
        "current_pass": "adapter_off",
    }
    save_json_atomic(state, output_dir / "state.json")

    pass_results_dir = output_dir / "pass_results"
    pass_results_dir.mkdir(exist_ok=True)

    state_path = output_dir / "state.json"

    # ---- Pass 1: adapter OFF (clean base model) ----
    results_off = _run_pass(
        sessions,
        model_name,
        adapter_path=None,
        results_dir=pass_results_dir,
        label="off",
        state=state,
        state_path=state_path,
    )

    state["current_pass"] = "cooldown"
    save_json_atomic(state, state_path)

    logger.info("Cooldown between passes...")
    wait_for_cooldown(45)

    # ---- Pass 2: adapter ON (base model + Test 8 adapter) ----
    state["current_pass"] = "adapter_on"
    state["completed_sessions"] = 0
    save_json_atomic(state, state_path)

    results_on = _run_pass(
        sessions,
        model_name,
        adapter_path=adapter_path,
        results_dir=pass_results_dir,
        label="on",
        state=state,
        state_path=state_path,
    )

    # ---- Combine results with grounding metrics ----
    state["current_pass"] = "aggregation"
    save_json_atomic(state, output_dir / "state.json")

    session_results = []
    for i, session in enumerate(sessions):
        if i >= len(results_off) or i >= len(results_on):
            break

        transcript = session["transcript"]
        grounding_off = {
            "entity": compute_entity_grounding(results_off[i], transcript),
            "triple": compute_triple_grounding(results_off[i], transcript),
        }
        grounding_on = {
            "entity": compute_entity_grounding(results_on[i], transcript),
            "triple": compute_triple_grounding(results_on[i], transcript),
        }
        comparison = compute_session_comparison(results_off[i], results_on[i])

        session_result = {
            "session_id": session["session_id"],
            "character": session["character"],
            "transcript_length": len(transcript),
            "adapter_off": results_off[i],
            "adapter_on": results_on[i],
            "grounding_off": grounding_off,
            "grounding_on": grounding_on,
            "comparison": comparison,
        }
        session_results.append(session_result)

    if not session_results:
        logger.warning("No session results collected")
        return

    # Save combined per-session results
    combined_dir = output_dir / "session_results"
    combined_dir.mkdir(exist_ok=True)
    for i, sr in enumerate(session_results):
        save_json_atomic(sr, combined_dir / f"session_{i + 1:03d}.json")

    # Aggregate
    aggregated = aggregate_results(session_results)
    save_json_atomic(aggregated, output_dir / "results.json")

    # Print summary
    a = aggregated.get("adapter_off", {})
    b = aggregated.get("adapter_on", {})
    c = aggregated.get("comparison", {})
    print(f"\n{'=' * 60}")
    print("  Test 11: Adapter-Active Extraction — Quality Comparison")
    print(f"  max_tokens={MAX_EXTRACTION_TOKENS}")
    print(f"{'=' * 60}")
    print(f"  Sessions:           {aggregated.get('total_sessions', 0)}")
    print(
        f"  Success rate:       OFF={a.get('success_rate', 0) * 100:.0f}%"
        f"   ON={b.get('success_rate', 0) * 100:.0f}%"
    )
    print(f"  Truncated:          OFF={a.get('truncated', 0)}   ON={b.get('truncated', 0)}")
    print(
        f"  Mean triples:       OFF={a.get('mean_triples', 0):.1f}"
        f"   ON={b.get('mean_triples', 0):.1f}"
    )
    print(
        f"  Entity grounding:   OFF={a.get('mean_entity_grounding', 0) * 100:.0f}%"
        f"   ON={b.get('mean_entity_grounding', 0) * 100:.0f}%"
    )
    print(
        f"  Triple grounding:   OFF={a.get('mean_triple_grounding', 0) * 100:.0f}%"
        f"   ON={b.get('mean_triple_grounding', 0) * 100:.0f}%"
    )
    print(f"  --- Both succeed ({c.get('both_success_count', 0)} sessions) ---")
    print(f"  Triple overlap:     {c.get('mean_triple_overlap', 0) * 100:.1f}%")
    print(f"  Predicate overlap:  {c.get('mean_predicate_overlap', 0) * 100:.1f}%")
    print(f"{'=' * 60}\n")

    state["current_pass"] = "complete"
    state["completed_sessions"] = len(session_results)
    save_json_atomic(state, output_dir / "state.json")
    logger.info("Results saved to %s", output_dir / "results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Test 11: Adapter-Active Extraction — Quality Comparison"
    )
    parser.add_argument(
        "--model",
        choices=list(BENCHMARK_MODELS.keys()),
        default="mistral",
        help="Model to use (default: mistral)",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=50,
        help="Number of sessions to evaluate (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest incomplete run",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Test 11: Adapter-Active Extraction — {args.model}")
    print(f"  Sessions: {args.num_sessions}, max_tokens: {MAX_EXTRACTION_TOKENS}")
    print(f"{'=' * 60}")

    run_experiment(
        model_name=args.model,
        num_sessions=args.num_sessions,
        resume=args.resume,
    )


if __name__ == "__main__":
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        main()
