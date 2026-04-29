"""Test 12: Session Cohort Smoke Test — Step 0 decision gate.

Validates whether rank-8 LoRA on small key cohorts (1, 2, 3, 5 keys) achieves
100% indexed-key recall after 30 epochs. This is the decision gate for the
multi-adapter session routing architecture.

Design:
  - For each cohort size N in [1, 2, 3, 5]:
    - Spawn a fresh Python subprocess for that cohort only.
    - The subprocess: loads model, creates adapter, trains 30 epochs, probes
      via smoke_test_adapter(), writes cohort_result.json, exits 0/1.
    - This subprocess isolation prevents WSL2 CUDA rapid-reload races:
      each sub-process loads the model at most twice (training load +
      smoke_test_adapter reload) in a brand-new process, which is the
      documented working pattern.
  - Cool down GPU between cohort subprocesses via gpu-cooldown.sh.
  - Outer orchestrator aggregates per-cohort JSONs into results.json.
  - Print a clear summary with pass/fail and triggered fall-back branch.

Two argparse modes (single file):
  Outer (default):    python experiments/test12_session_cohort_smoke.py
  Inner (subprocess): python experiments/test12_session_cohort_smoke.py
                          --cohort N --output-dir DIR

Fall-back branches:
  - All pass      -> architecture proceeds as planned (no fall-back)
  - N=1 or N=2 fail -> Fall-back A: bundle >= 3 keys before training
  - N=3 fails     -> Fall-back B: weekly cadence, 2-3 adapters total
  - N=5 fails     -> STOP: investigate Test 4b regression, do not proceed

Bug fixes vs prior version:
  1. Subprocess-per-cohort isolation — eliminates WSL2 CUDA rapid-reload race.
     The prior version did training + smoke_test_adapter (which calls
     from_pretrained) in the same process, causing CUDA driver errors on
     the third model load inside one process.
  2. target_modules now includes all four projection matrices
     ["q_proj", "v_proj", "k_proj", "o_proj"] to match production episodic
     adapter in configs/default.yaml.

Estimated wall-clock: 30-45 min (single model, Mistral NF4).

Usage:
    python experiments/test12_session_cohort_smoke.py
    python experiments/test12_session_cohort_smoke.py --model mistral
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

COHORT_SIZES = [1, 2, 3, 5]
EPOCHS = 30
ADAPTER_RANK = 8
ADAPTER_ALPHA = 16  # 2 * rank
ADAPTER_LR = 1e-4
# Production target modules — must match configs/default.yaml adapters.episodic.target_modules
ADAPTER_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
PASS_THRESHOLD = 1.0  # 100% recall required

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    model_output_dir,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "outputs" / "test12_session_cohort_smoke"

# ---------------------------------------------------------------------------
# Synthetic QA factory — fictional people, no real personal data
# ---------------------------------------------------------------------------

_SYNTHETIC_TEMPLATES = [
    {
        "question": "What is {name}'s occupation?",
        "answer": "{name} is a {job}.",
    },
    {
        "question": "Where does {name} live?",
        "answer": "{name} lives in {city}.",
    },
    {
        "question": "What is {name}'s favourite hobby?",
        "answer": "{name}'s favourite hobby is {hobby}.",
    },
    {
        "question": "How old is {name}?",
        "answer": "{name} is {age} years old.",
    },
    {
        "question": "What language does {name} speak?",
        "answer": "{name} speaks {language}.",
    },
]

_PEOPLE = [
    {
        "name": "Aria Chen",
        "job": "software engineer",
        "city": "Vienna",
        "hobby": "painting",
        "age": "34",
        "language": "Mandarin",
    },
    {
        "name": "Bence Toth",
        "job": "marine biologist",
        "city": "Budapest",
        "hobby": "chess",
        "age": "28",
        "language": "Hungarian",
    },
    {
        "name": "Cordelia Frost",
        "job": "astrophysicist",
        "city": "Edinburgh",
        "hobby": "hiking",
        "age": "41",
        "language": "French",
    },
    {
        "name": "Daisuke Mori",
        "job": "architect",
        "city": "Osaka",
        "hobby": "origami",
        "age": "37",
        "language": "Japanese",
    },
    {
        "name": "Elif Kaya",
        "job": "veterinarian",
        "city": "Istanbul",
        "hobby": "cycling",
        "age": "30",
        "language": "Turkish",
    },
]


def generate_synthetic_qa(n: int) -> list[dict]:
    """Generate N distinct synthetic QA pairs from fictional people + fact templates.

    Iterates over (person x template) pairs in order, so every combination is
    unique. Raises ValueError if n exceeds the available pool.

    Args:
        n: Number of QA pairs to generate.

    Returns:
        List of {"question": str, "answer": str} dicts.
    """
    pool = []
    for person in _PEOPLE:
        for tmpl in _SYNTHETIC_TEMPLATES:
            pool.append(
                {
                    "question": tmpl["question"].format(**person),
                    "answer": tmpl["answer"].format(**person),
                }
            )

    if n > len(pool):
        raise ValueError(
            f"Requested {n} QA pairs but pool has only {len(pool)}. Add more people or templates."
        )
    return pool[:n]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def wait_for_cooldown(target: int = 52) -> None:
    """Block until GPU temperature drops below target Celsius.

    Falls back to a 60-second sleep if the cooldown script is unavailable.

    Args:
        target: Target GPU temperature in Celsius.
    """
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
        logger.info("GPU cooled to <= %d C", target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Cooldown script failed (%s), falling back to 60s sleep", e)
        time.sleep(60)


def save_json_atomic(data: dict, target: Path) -> None:
    """Write JSON atomically via temp file + rename.

    Args:
        data: Data to serialise as JSON.
        target: Destination path for the JSON file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(tmp_path).replace(target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def check_disk_space(path: Path, min_gb: float = 10.0) -> None:
    """Raise RuntimeError if free disk space is below min_gb.

    Args:
        path: Path to check disk usage for.
        min_gb: Minimum required free space in gigabytes.
    """
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free (need {min_gb} GB)")
    logger.info("Disk: %.1f GB free", free_gb)


# ---------------------------------------------------------------------------
# Fall-back branch logic
# ---------------------------------------------------------------------------


def interpret_results(cohort_results: list[dict]) -> dict:
    """Map per-cohort pass/fail to the decision-gate fall-back branch.

    Args:
        cohort_results: List of per-cohort result dicts with "n_keys" and "passed".

    Returns:
        Dict with "branch", "label", and "description" keys.
    """
    pass_by_n = {r["n_keys"]: r["passed"] for r in cohort_results}

    if all(pass_by_n.values()):
        return {
            "branch": "none",
            "label": "ALL PASS",
            "description": (
                "All cohorts passed. Architecture proceeds as planned. No fall-back required."
            ),
        }

    if not pass_by_n.get(5, True):
        return {
            "branch": "stop",
            "label": "STOP",
            "description": (
                "N=5 cohort failed. Test 4b already validated this cohort. "
                "Investigate before any further architecture work — something has changed."
            ),
        }

    if not pass_by_n.get(3, True):
        return {
            "branch": "B",
            "label": "Fall-back B: weekly cadence",
            "description": (
                "N=3 cohort failed. Drop per-day session adapter. Train one or two "
                "session adapters per week instead. Recall delay is hours-to-days, "
                "not minutes."
            ),
        }

    # N=1 or N=2 failed (N=3 and N=5 passed)
    return {
        "branch": "A",
        "label": "Fall-back A: bundle >= 3 keys",
        "description": (
            "N=1 or N=2 cohort failed. Step 6 must queue conversations and only "
            "trigger session training when cumulative new-key count >= 3 OR a "
            "30-minute timer fires (whichever first). Minute-1 recall is no longer "
            "guaranteed; recall lands on bundle-flush or timer."
        ),
    }


# ---------------------------------------------------------------------------
# INNER MODE — one cohort, one fresh process
# ---------------------------------------------------------------------------


def run_cohort_inner(n_keys: int, output_dir: Path, model_name: str) -> None:
    """Run a single cohort in the current (fresh) subprocess.

    This function is the entry point for inner mode (--cohort N --output-dir DIR).
    It performs:
      1. Load base model (first from_pretrained call).
      2. Create fresh LoRA adapter with production target_modules.
      3. Train 30 epochs with constant LR, no warmup, no weight decay.
      4. Save adapter + keyed_pairs.json + simhash_registry.json.
      5. Call smoke_test_adapter() which does a second from_pretrained internally.
      6. Write cohort_result.json to output_dir.
      7. Exit with code 0 on success, 1 on failure.

    The two-load pattern (this load + smoke_test reload) is the documented
    working limit for WSL2 CUDA driver stability.

    Args:
        n_keys: Cohort size (number of synthetic QA pairs to train).
        output_dir: Directory for all artefacts. Must already exist or be
            creatable. Writes: adapter/<adapter_name>/, keyed_pairs.json,
            simhash_registry.json, cohort_result.json.
        model_name: Key into BENCHMARK_MODELS (e.g. "mistral").
    """
    import gc

    import torch

    from experiments.utils.test_harness import (
        IndexedDataset,
        smoke_test_adapter,
    )
    from paramem.models.loader import create_adapter, load_base_model
    from paramem.training.indexed_memory import (
        assign_keys,
        build_registry,
        format_indexed_training,
        save_registry,
    )
    from paramem.training.trainer import train_adapter
    from paramem.utils.config import AdapterConfig, TrainingConfig

    adapter_name = f"smoke_n{n_keys}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = BENCHMARK_MODELS[model_name]

    logger.info(
        "Inner cohort n=%d: adapter=%s, target_modules=%s",
        n_keys,
        adapter_name,
        ADAPTER_TARGET_MODULES,
    )

    # Step 1: Generate synthetic QA pairs — no distillation, no LLM call
    qa_pairs = generate_synthetic_qa(n_keys)

    # Step 2: Assign indexed keys and build SimHash registry
    keyed_pairs = assign_keys(qa_pairs, start_index=1)
    registry = build_registry(keyed_pairs)

    # Persist artefacts before training so smoke_test_adapter can find them
    save_json_atomic(
        [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in keyed_pairs
        ],
        output_dir / "keyed_pairs.json",
    )
    save_registry(registry, output_dir / "simhash_registry.json")

    # Step 3: Load base model — first from_pretrained in this process
    logger.info("Loading base model: %s", model_config.model_id)
    model, tokenizer = load_base_model(model_config)

    # Step 4: Create fresh adapter — four production target_modules
    adapter_config = AdapterConfig(
        rank=ADAPTER_RANK,
        alpha=ADAPTER_ALPHA,
        learning_rate=ADAPTER_LR,
        target_modules=ADAPTER_TARGET_MODULES,
        dropout=0.05,
    )
    model = create_adapter(model, adapter_config, adapter_name)

    # Step 5: Build tokenised dataset
    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    # Step 6: Configure training — constant LR, no warmup, no weight decay
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=EPOCHS,
        warmup_ratio=0.0,
        warmup_steps=0,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="no",
        save_total_limit=0,
        early_stopping=False,
    )

    adapter_output_dir = output_dir / "adapter"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training cohort n=%d: adapter=%s, epochs=%d, lr=%s, scheduler=constant",
        n_keys,
        adapter_name,
        EPOCHS,
        ADAPTER_LR,
    )

    t0 = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=adapter_output_dir,
        run_name=f"smoke-{adapter_name}",
    )
    train_time = time.time() - t0

    logger.info(
        "Cohort n=%d trained in %.1fs, loss=%.4f",
        n_keys,
        train_time,
        metrics.get("train_loss", -1),
    )

    # Release the training model before smoke_test_adapter does its own load.
    # This keeps peak VRAM usage bounded — the probe load is the second
    # from_pretrained in this process (the working WSL2 pattern).
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Step 7: Probe recall — second from_pretrained via smoke_test_adapter
    logger.info("Probing recall for cohort n=%d via smoke_test_adapter", n_keys)
    probe_result = smoke_test_adapter(output_dir, model_name, adapter_name=adapter_name)

    passed = probe_result["rate"] >= PASS_THRESHOLD
    status_str = "PASS" if passed else "FAIL"

    logger.info(
        "Cohort n=%d %s: %d/%d (%.0f%%), conf=%.3f",
        n_keys,
        status_str,
        probe_result["exact_count"],
        probe_result["total"],
        probe_result["rate"] * 100,
        probe_result["mean_confidence"],
    )

    cohort_record = {
        "n_keys": n_keys,
        "adapter_name": adapter_name,
        "output_dir": str(output_dir),
        "passed": passed,
        "recall_rate": probe_result["rate"],
        "exact_count": probe_result["exact_count"],
        "total": probe_result["total"],
        "mean_confidence": probe_result["mean_confidence"],
        "per_key": probe_result.get("per_key", []),
        "train_loss": metrics.get("train_loss"),
        "train_time_s": round(train_time, 1),
        "target_modules": ADAPTER_TARGET_MODULES,
    }

    save_json_atomic(cohort_record, output_dir / "cohort_result.json")
    logger.info("cohort_result.json written to %s", output_dir)

    sys.exit(0 if passed else 1)


# ---------------------------------------------------------------------------
# OUTER MODE — orchestrator, spawns one subprocess per cohort
# ---------------------------------------------------------------------------


def run_experiment_outer(model_name: str) -> None:
    """Outer orchestrator: spawn one fresh subprocess per cohort, aggregate results.

    Each cohort runs in its own process to prevent WSL2 CUDA rapid-reload
    races. The orchestrator:
      1. Creates a timestamped output directory.
      2. For each cohort size N in COHORT_SIZES (skipping already-done):
         a. Calls wait_for_cooldown() before spawning.
         b. Spawns subprocess: sys.executable __file__ --cohort N
                --output-dir <cohort_dir> --model <model_name>
         c. Reads cohort_result.json produced by the subprocess.
         d. Appends result to aggregated list and persists results.json.
      3. Writes final decision-gate interpretation to results.json.

    Args:
        model_name: Key into BENCHMARK_MODELS (e.g. "mistral").
    """
    if model_name not in BENCHMARK_MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(BENCHMARK_MODELS.keys())}")

    model_config = BENCHMARK_MODELS[model_name]
    output_dir = model_output_dir(OUTPUT_BASE, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_disk_space(output_dir, min_gb=10.0)

    results_path = output_dir / "results.json"

    # Load existing partial results so a restart does not re-run completed cohorts
    partial: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            partial = json.load(f)
        logger.info("Resuming: found %d completed cohorts", len(partial.get("cohorts", [])))

    completed_ns = {r["n_keys"] for r in partial.get("cohorts", [])}

    run_config = {
        "model": model_name,
        "model_id": model_config.model_id,
        "cohort_sizes": COHORT_SIZES,
        "epochs": EPOCHS,
        "adapter_rank": ADAPTER_RANK,
        "adapter_alpha": ADAPTER_ALPHA,
        "adapter_lr": ADAPTER_LR,
        "adapter_target_modules": ADAPTER_TARGET_MODULES,
        "lr_scheduler": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "pass_threshold": PASS_THRESHOLD,
    }

    all_cohort_results = list(partial.get("cohorts", []))

    print(f"\n{'=' * 68}")
    print(f"  Test 12: Session Cohort Smoke Test — {model_name}")
    print(f"  Cohorts: {COHORT_SIZES}, {EPOCHS} epochs, rank={ADAPTER_RANK}")
    print(f"  target_modules: {ADAPTER_TARGET_MODULES}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 68}")

    for i, n in enumerate(COHORT_SIZES):
        if n in completed_ns:
            logger.info("Cohort n=%d already complete — skipping", n)
            continue

        cohort_dir = output_dir / f"cohort_n{n}"
        cohort_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Cohort n=%d: cooling down GPU before subprocess ===", n)
        wait_for_cooldown(52)

        print(f"\n--- Cohort n={n}: launching subprocess ---")

        cohort_t0 = time.time()

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cohort",
            str(n),
            "--output-dir",
            str(cohort_dir),
            "--model",
            model_name,
        ]
        logger.info("Subprocess cmd: %s", " ".join(cmd))

        # capture_output=False: let stdout/stderr stream directly so the user
        # can see training progress. Never pipe through head/tail (SIGPIPE risk).
        proc = subprocess.run(cmd, check=False, capture_output=False)

        cohort_elapsed = time.time() - cohort_t0
        cohort_result_path = cohort_dir / "cohort_result.json"

        if not cohort_result_path.exists():
            logger.error(
                "Cohort n=%d subprocess exited (rc=%d) without writing cohort_result.json",
                n,
                proc.returncode,
            )
            # Record a failed sentinel so subsequent runs skip this cohort
            # rather than re-triggering another potentially-crashy subprocess.
            cohort_record = {
                "n_keys": n,
                "adapter_name": f"smoke_n{n}",
                "output_dir": str(cohort_dir),
                "passed": False,
                "recall_rate": 0.0,
                "exact_count": 0,
                "total": n,
                "mean_confidence": 0.0,
                "per_key": [],
                "train_loss": None,
                "train_time_s": None,
                "target_modules": ADAPTER_TARGET_MODULES,
                "subprocess_rc": proc.returncode,
                "error": "subprocess exited without writing cohort_result.json",
                "cohort_wall_time_s": round(cohort_elapsed, 1),
            }
        else:
            with open(cohort_result_path) as f:
                cohort_record = json.load(f)
            cohort_record["subprocess_rc"] = proc.returncode
            cohort_record["cohort_wall_time_s"] = round(cohort_elapsed, 1)

        all_cohort_results.append(cohort_record)

        # Persist after every cohort — crash in a later cohort must not lose this
        save_json_atomic(
            {"run_config": run_config, "cohorts": all_cohort_results},
            results_path,
        )

        passed = cohort_record.get("passed", False)
        status_str = "PASS" if passed else "FAIL"
        print(
            f"  n={n}: {status_str} — "
            f"{cohort_record.get('exact_count', 0)}/{cohort_record.get('total', n)} "
            f"({cohort_record.get('recall_rate', 0.0) * 100:.0f}%) "
            f"conf={cohort_record.get('mean_confidence', 0.0):.3f} "
            f"loss={cohort_record.get('train_loss') or -1:.4f} "
            f"subprocess_rc={proc.returncode}"
        )

    # Reload final results (may include earlier partial run entries)
    with open(results_path) as f:
        final_results = json.load(f)

    decision = interpret_results(final_results["cohorts"])
    final_results["decision"] = decision
    save_json_atomic(final_results, results_path)

    # Print summary
    print(f"\n{'=' * 68}")
    print("  Test 12: Session Cohort Smoke Test — Summary")
    print(f"{'=' * 68}")
    for r in final_results["cohorts"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  n={r['n_keys']:>2}: {status}  "
            f"{r['exact_count']}/{r['total']}  "
            f"conf={r['mean_confidence']:.3f}  "
            f"loss={r.get('train_loss') or -1:.4f}  "
            f"train={r.get('train_time_s', 0) or 0:.0f}s"
        )
    print(f"\n  Decision: [{decision['label']}]")
    print(f"  {decision['description']}")
    print(f"\n  Results saved to: {results_path}")
    print(f"{'=' * 68}\n")

    logger.info("Test 12 complete. Results: %s", results_path)


# ---------------------------------------------------------------------------
# Entry point — outer orchestrator or inner cohort worker
# ---------------------------------------------------------------------------


def main() -> None:
    """Dispatch to outer orchestrator or inner cohort worker based on argparse.

    Outer mode (no --cohort):
        Iterates cohort sizes, spawns one subprocess per cohort, aggregates
        results into results.json. Calls acquire_gpu() before orchestrating.

    Inner mode (--cohort N --output-dir DIR):
        Runs one cohort in the current process. Calls acquire_gpu() before
        any GPU work. Writes cohort_result.json. Exits 0/1.
    """
    parser = argparse.ArgumentParser(
        description="Test 12: Session Cohort Smoke Test — Step 0 decision gate"
    )
    parser.add_argument(
        "--model",
        choices=list(BENCHMARK_MODELS.keys()),
        default="mistral",
        help="Model to test (default: mistral)",
    )
    # Inner-mode flags — not used in outer mode
    parser.add_argument(
        "--cohort",
        type=int,
        default=None,
        metavar="N",
        help="(Inner mode) cohort size to run in this subprocess",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(Inner mode) directory to write cohort_result.json and adapter artefacts",
    )
    args = parser.parse_args()

    if args.cohort is not None:
        # ----- INNER MODE -----
        if args.output_dir is None:
            parser.error("--output-dir is required in inner mode (--cohort N)")
        run_cohort_inner(
            n_keys=args.cohort,
            output_dir=args.output_dir,
            model_name=args.model,
        )
    else:
        # ----- OUTER MODE -----
        print(f"\n{'=' * 68}")
        print("  Test 12: Session Cohort Smoke Test")
        print(f"  Model: {args.model}  Cohorts: {COHORT_SIZES}  Epochs: {EPOCHS}")
        print(f"  target_modules: {ADAPTER_TARGET_MODULES}")
        print(f"{'=' * 68}")
        run_experiment_outer(model_name=args.model)


if __name__ == "__main__":
    from experiments.utils.gpu_guard import acquire_gpu

    with acquire_gpu():
        main()
