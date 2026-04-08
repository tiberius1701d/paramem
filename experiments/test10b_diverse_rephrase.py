"""Test 10b: Diverse Rephrasing Probe.

Standalone evaluation script that probes existing Test 10 adapter checkpoints
with genuinely diverse question variations (colloquial, indirect, partial,
contextual, formal-alternative) instead of just passive voice transformations.

Uses two scoring methods:
  - Entity match (strict): case-insensitive substring, comparable to Test 10
  - LLM-as-judge (relaxed): base model judges semantic correctness

No training — pure evaluation pass over existing checkpoints.

Usage:
    python experiments/test10b_diverse_rephrase.py --model mistral
    python experiments/test10b_diverse_rephrase.py --model mistral \\
        --run-dir outputs/test10_grokking/mistral/20260403_164507
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

from peft import PeftModel  # noqa: E402

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    model_output_dir,
    setup_logging,
)
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    load_adapter,
    load_base_model,
)

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "outputs" / "test10b_diverse_rephrase"
TEST10_BASE = project_root / "outputs" / "test10_grokking"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
RESULTS_FILENAME = "diverse_rephrase_results.json"
QUESTIONS_FILENAME = "diverse_rephrase_questions.json"
SUMMARY_FILENAME = "diverse_rephrase_summary.json"

# Five diverse rephrasing styles with few-shot examples.
# Temperature=0.0 — diversity comes from prompt variety, not sampling.
REPHRASE_STYLES = {
    "colloquial": {
        "system": (
            "Rephrase the following question in casual, everyday language. "
            "Use informal phrasing like you'd ask a friend. "
            "The rephrased question must have the same answer. "
            "Output ONLY the rephrased question."
        ),
        "example_in": "What did the audience enjoy?",
        "example_out": "So what was it the audience was really into?",
    },
    "indirect": {
        "system": (
            "Rephrase the following question as an indirect question or request. "
            "Use phrasing like 'I was wondering...', 'Could you tell me...', "
            "'Do you happen to know...'. "
            "The rephrased question must have the same answer. "
            "Output ONLY the rephrased question."
        ),
        "example_in": "What did the audience enjoy?",
        "example_out": "Could you tell me what it was the audience found enjoyable?",
    },
    "partial": {
        "system": (
            "Rephrase the following question by approaching it from a different "
            "angle — mention the object or context first, then ask about the "
            "relationship. Use a different sentence structure entirely. "
            "The rephrased question must have the same answer. "
            "Output ONLY the rephrased question."
        ),
        "example_in": "What did the audience enjoy?",
        "example_out": "Which performance received a positive reception from the audience?",
    },
    "contextual": {
        "system": (
            "Rephrase the following question by adding a brief contextual "
            "lead-in before the actual question. Frame it as part of a "
            "conversation about the topic. "
            "The rephrased question must have the same answer. "
            "Output ONLY the rephrased question."
        ),
        "example_in": "What did the audience enjoy?",
        "example_out": "Speaking of the event, what was it that the audience particularly enjoyed?",
    },
    "formal_alternative": {
        "system": (
            "Rephrase the following question using formal, academic-style "
            "phrasing with a completely different sentence structure. "
            "Use nominalization or embedded clauses. "
            "The rephrased question must have the same answer. "
            "Output ONLY the rephrased question."
        ),
        "example_in": "What did the audience enjoy?",
        "example_out": (
            "What form of entertainment was received most favorably by those in attendance?"
        ),
    },
}


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


def entity_match(generated: str, expected_entity: str) -> bool:
    """Case-insensitive substring match for entity in generated text."""
    if not expected_entity:
        return False
    return expected_entity.lower() in generated.lower()


# ============================================================================
# Run directory and checkpoint discovery
# ============================================================================


def find_test10_run_dir(model_name: str, run_dir_override: str | None = None) -> Path:
    """Find the Test 10 run directory (input source) for this model."""
    if run_dir_override:
        p = Path(run_dir_override)
        if p.exists() and (p / "state.json").exists():
            return p
        raise FileNotFoundError(f"Run dir not found or missing state.json: {p}")

    model_dir = TEST10_BASE / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No Test 10 output for model {model_name}: {model_dir}")

    # Find latest run with state.json
    subdirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and (d / "state.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No completed Test 10 runs in {model_dir}")
    return subdirs[0]


def discover_checkpoints(run_dir: Path) -> list[Path]:
    """Find all epoch directories with saved adapters, sorted numerically."""
    checkpoints = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("epoch_") and (d / "adapter").exists():
            try:
                epoch_num = int(d.name.split("_")[1])
                checkpoints.append((epoch_num, d))
            except ValueError:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return [d for _, d in checkpoints]


# ============================================================================
# Diverse question generation
# ============================================================================


def generate_diverse_questions(
    keyed_pairs: list[dict],
    model,
    tokenizer,
    cache_path: Path,
) -> list[dict]:
    """Generate 5 diverse rephrasings per QA pair using style prompts.

    Cached to disk. Regenerated only if cache is missing.
    Uses base model (adapter OFF), temperature=0.0.
    """
    if cache_path.exists():
        logger.info("Loading cached diverse questions from %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    logger.info(
        "Generating diverse questions: %d pairs x %d styles", len(keyed_pairs), len(REPHRASE_STYLES)
    )
    questions = []

    for i, kp in enumerate(keyed_pairs):
        original_q = kp["question"]
        expected_a = kp["answer"]
        expected_entity = kp.get("source_object", "")

        for style_name, style_config in REPHRASE_STYLES.items():
            messages = [
                {"role": "system", "content": style_config["system"]},
                {"role": "user", "content": style_config["example_in"]},
                {"role": "assistant", "content": style_config["example_out"]},
                {"role": "user", "content": original_q},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    rephrased = generate_answer(
                        model, tokenizer, prompt, max_new_tokens=100, temperature=0.0
                    )
            else:
                rephrased = generate_answer(
                    model, tokenizer, prompt, max_new_tokens=100, temperature=0.0
                )

            rephrased = rephrased.strip()
            if not rephrased or rephrased == original_q:
                rephrased = original_q

            questions.append(
                {
                    "key": kp["key"],
                    "style": style_name,
                    "original_question": original_q,
                    "rephrased_question": rephrased,
                    "expected_answer": expected_a,
                    "expected_entity": expected_entity,
                }
            )

        if (i + 1) % 20 == 0:
            logger.info(
                "  Generated %d/%d pairs (%d questions)", i + 1, len(keyed_pairs), len(questions)
            )

    logger.info("Generated %d diverse questions total", len(questions))
    save_json_atomic(questions, cache_path)
    return questions


# ============================================================================
# LLM-as-judge scoring
# ============================================================================


def judge_answer(
    model,
    tokenizer,
    question: str,
    expected_answer: str,
    generated_answer: str,
) -> bool:
    """LLM-as-judge: does the generated answer correctly convey the expected fact?

    Uses base model (adapter OFF), temperature=0.0.
    Returns True if the model judges the answer as correct.
    """
    judge_prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are an answer correctness judge. Determine if the response "
                "correctly conveys the expected fact. Answer ONLY 'Yes' or 'No'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Expected fact: {expected_answer}\n"
                f"Response: {generated_answer}\n\n"
                "Does the response correctly convey the expected fact? Yes or No?"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        judge_prompt_messages, tokenize=False, add_generation_prompt=True
    )

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            verdict = generate_answer(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)
    else:
        verdict = generate_answer(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)

    return verdict.strip().lower().startswith("yes")


# ============================================================================
# Probing
# ============================================================================


def probe_diverse_rephrase(
    model,
    tokenizer,
    diverse_questions: list[dict],
) -> dict:
    """Probe adapter with all diverse questions, scoring with entity match + judge."""
    results = []

    for i, dq in enumerate(diverse_questions):
        # Generate answer with adapter ON
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": dq["rephrased_question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0)

        # Strict: entity match
        strict_match = entity_match(generated, dq["expected_entity"])

        # Relaxed: LLM-as-judge (adapter OFF)
        judge_match = judge_answer(
            model,
            tokenizer,
            dq["rephrased_question"],
            dq["expected_answer"],
            generated,
        )

        results.append(
            {
                "key": dq["key"],
                "style": dq["style"],
                "rephrased_question": dq["rephrased_question"],
                "original_question": dq["original_question"],
                "expected_entity": dq["expected_entity"],
                "expected_answer": dq["expected_answer"],
                "generated": generated,
                "entity_match": strict_match,
                "judge_match": judge_match,
            }
        )

        if (i + 1) % 50 == 0:
            em_so_far = sum(1 for r in results if r["entity_match"])
            jm_so_far = sum(1 for r in results if r["judge_match"])
            logger.info(
                "  Probed %d/%d: entity_match=%d (%.1f%%), judge=%d (%.1f%%)",
                i + 1,
                len(diverse_questions),
                em_so_far,
                em_so_far / (i + 1) * 100,
                jm_so_far,
                jm_so_far / (i + 1) * 100,
            )

    # Aggregate
    total = len(results)
    entity_matched = sum(1 for r in results if r["entity_match"])
    judge_matched = sum(1 for r in results if r["judge_match"])

    # Per-style breakdown
    per_style = {}
    for style_name in REPHRASE_STYLES:
        style_results = [r for r in results if r["style"] == style_name]
        style_total = len(style_results)
        per_style[style_name] = {
            "total": style_total,
            "entity_match_count": sum(1 for r in style_results if r["entity_match"]),
            "entity_match_rate": round(
                sum(1 for r in style_results if r["entity_match"]) / style_total, 4
            )
            if style_total > 0
            else 0.0,
            "judge_match_count": sum(1 for r in style_results if r["judge_match"]),
            "judge_match_rate": round(
                sum(1 for r in style_results if r["judge_match"]) / style_total, 4
            )
            if style_total > 0
            else 0.0,
        }

    return {
        "total": total,
        "entity_match_count": entity_matched,
        "entity_match_rate": round(entity_matched / total, 4) if total > 0 else 0.0,
        "judge_match_count": judge_matched,
        "judge_match_rate": round(judge_matched / total, 4) if total > 0 else 0.0,
        "per_style": per_style,
        "results": results,
    }


# ============================================================================
# Checkpoint evaluation
# ============================================================================


def build_summary(run_dir: Path, checkpoints: list[Path]) -> dict:
    """Aggregate results from all checkpoints into a summary table."""
    rows = []
    for cp_dir in checkpoints:
        results_path = cp_dir / RESULTS_FILENAME
        if not results_path.exists():
            continue
        with open(results_path) as f:
            data = json.load(f)
        row = {
            "epoch": data.get("epoch", int(cp_dir.name.split("_")[1])),
            "entity_match_rate": data["entity_match_rate"],
            "judge_match_rate": data["judge_match_rate"],
        }
        for style_name, style_data in data.get("per_style", {}).items():
            row[f"entity_{style_name}"] = style_data["entity_match_rate"]
            row[f"judge_{style_name}"] = style_data["judge_match_rate"]
        rows.append(row)

    rows.sort(key=lambda r: r["epoch"])
    summary = {"checkpoints": rows}
    save_json_atomic(summary, run_dir / SUMMARY_FILENAME)
    logger.info("Summary saved to %s (%d checkpoints)", SUMMARY_FILENAME, len(rows))
    return summary


# ============================================================================
# Main
# ============================================================================


def run_test10b(
    model_name: str,
    run_dir_override: str | None = None,
    resume: bool = False,
):
    """Main orchestration."""
    # Input: Test 10 run directory (adapters + keyed_pairs)
    test10_dir = find_test10_run_dir(model_name, run_dir_override)
    logger.info("Test 10 input dir: %s", test10_dir)

    # Output: Test 10b own directory
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
    logger.info("Output dir: %s", output_dir)

    # Load keyed pairs from Test 10
    kp_path = test10_dir / "keyed_pairs.json"
    if not kp_path.exists():
        raise FileNotFoundError(f"keyed_pairs.json not found in {test10_dir}")
    with open(kp_path) as f:
        keyed_pairs = json.load(f)
    logger.info("Loaded %d keyed pairs", len(keyed_pairs))

    # Check disk space
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    if free_gb < 20:
        logger.error("Insufficient disk space: %.1f GB free", free_gb)
        return
    logger.info("Disk: %.1f GB free", free_gb)

    # Discover checkpoints from Test 10
    checkpoints = discover_checkpoints(test10_dir)
    logger.info("Found %d checkpoints to evaluate", len(checkpoints))

    if not checkpoints:
        logger.warning("No checkpoints found in %s", test10_dir)
        return

    # Write state for tstatus discovery
    state = {
        "test10_input": str(test10_dir),
        "total_checkpoints": len(checkpoints),
        "completed_checkpoints": 0,
        "model": model_name,
    }
    save_json_atomic(state, output_dir / "state.json")

    # Save run config
    save_json_atomic(
        {
            "model": model_name,
            "test10_input": str(test10_dir),
            "checkpoints": len(checkpoints),
        },
        output_dir / "run_config.json",
    )

    # Load model
    model_config = BENCHMARK_MODELS[model_name]
    logger.info("Loading model: %s", model_name)
    model, tokenizer = load_base_model(model_config)
    model.gradient_checkpointing_disable()

    # Generate diverse questions (cached in output dir)
    cache_path = output_dir / QUESTIONS_FILENAME
    diverse_questions = generate_diverse_questions(keyed_pairs, model, tokenizer, cache_path)
    logger.info("Diverse questions ready: %d total", len(diverse_questions))

    # Warn about keys with empty expected_entity
    empty_entity = sum(1 for dq in diverse_questions if not dq.get("expected_entity"))
    if empty_entity:
        logger.warning(
            "%d/%d questions have empty expected_entity (entity_match will be False)",
            empty_entity,
            len(diverse_questions),
        )

    # Cooldown after question generation (sustained GPU burst)
    wait_for_cooldown(45)

    # Evaluate each checkpoint
    completed = 0
    for i, cp_dir in enumerate(checkpoints):
        if is_paused():
            logger.info("Paused — stopping after %d checkpoints", i)
            break

        epoch_name = cp_dir.name  # e.g. "epoch_030"
        result_dir = output_dir / epoch_name
        result_dir.mkdir(exist_ok=True)

        # Skip if already evaluated
        if (result_dir / RESULTS_FILENAME).exists():
            logger.info("  Skipping %s (results exist)", epoch_name)
            completed += 1
            continue

        logger.info("Checkpoint %d/%d: %s", i + 1, len(checkpoints), epoch_name)

        # Load adapter from Test 10's checkpoint
        if isinstance(model, PeftModel):
            model = model.base_model.model
        adapter_path = cp_dir / "adapter"
        model = load_adapter(model, str(adapter_path), "episodic")
        model.gradient_checkpointing_disable()

        # Probe
        logger.info("  Probing %d diverse questions...", len(diverse_questions))
        probe_result = probe_diverse_rephrase(model, tokenizer, diverse_questions)
        epoch_num = int(epoch_name.split("_")[1])
        probe_result["epoch"] = epoch_num

        save_json_atomic(probe_result, result_dir / RESULTS_FILENAME)
        completed += 1
        logger.info(
            "  E%d: entity_match=%.1f%%, judge=%.1f%%",
            epoch_num,
            probe_result["entity_match_rate"] * 100,
            probe_result["judge_match_rate"] * 100,
        )

        # Update state
        state["completed_checkpoints"] = completed
        save_json_atomic(state, output_dir / "state.json")

        # Cooldown between checkpoints
        if i < len(checkpoints) - 1:
            if isinstance(model, PeftModel):
                model = model.base_model.model
            wait_for_cooldown(45)

    # Unwrap final adapter
    if isinstance(model, PeftModel):
        model = model.base_model.model

    # Build summary from output dir
    output_checkpoints = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    summary = build_summary(output_dir, output_checkpoints)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("  Test 10b: Diverse Rephrasing Probe — Summary")
    print(f"{'=' * 80}")
    print(f"  {'Epoch':>5} | {'Entity%':>7} | {'Judge%':>7} | ", end="")
    for style in REPHRASE_STYLES:
        print(f"{style[:6]:>7} ", end="")
    print()
    print(f"  {'-' * 5}-+-{'-' * 7}-+-{'-' * 7}-+-", end="")
    for _ in REPHRASE_STYLES:
        print(f"{'-' * 7}-", end="")
    print()
    for row in summary.get("checkpoints", []):
        print(
            f"  {row['epoch']:>5} | "
            f"{row['entity_match_rate'] * 100:>6.1f}% | "
            f"{row['judge_match_rate'] * 100:>6.1f}% | ",
            end="",
        )
        for style in REPHRASE_STYLES:
            rate = row.get(f"entity_{style}", 0)
            print(f"{rate * 100:>6.1f}% ", end="")
        print()
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Test 10b: Diverse Rephrasing Probe")
    parser.add_argument(
        "--model",
        choices=list(BENCHMARK_MODELS.keys()),
        default="mistral",
        help="Model to use (default: mistral)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override: path to Test 10 run directory (input)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest incomplete run",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Test 10b: Diverse Rephrasing Probe — {args.model}")
    print(f"{'=' * 60}")

    run_test10b(
        model_name=args.model,
        run_dir_override=args.run_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        main()
