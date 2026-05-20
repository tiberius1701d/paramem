"""Shared experiment infrastructure for extended evaluation tests.

Wraps the existing indexed key pipeline into reusable functions
for consistent experiment setup across all 7 tests.

Environment loading is a per-script concern — call
:func:`load_test_env` from a script's main() / argparse entrypoint
when you need ``.env`` populated. Module-level ``load_dotenv`` was
removed on 2026-04-28 because importing this module from production
code paths (``ConsolidationLoop._run_recall_sanity_probe``) re-set
operator env vars on first import, defeating any caller that had
explicitly cleared a key — e.g. the smoke harness running under
Security OFF saw ``PARAMEM_DAILY_PASSPHRASE`` snap back from disk
mid-run, with subsequent saves silently encrypted under a "popped"
identity.

QA-shape harness functions (distill_qa_pairs, distill_session,
train_indexed_keys, evaluate_indexed_recall, evaluate_individual_qa,
smoke_test_adapter) were retired on 2026-05-20 to
:mod:`archive.experiments.legacy_harness`.  Live tests use the
entry-format evaluation path via
:func:`paramem.training.recall_eval.evaluate_indexed_recall` directly.
"""

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402


def load_test_env() -> None:
    """Source ``.env`` into the current process and set the CUDA alloc default.

    Call from a script's main() / CLI entrypoint, NOT at module scope.
    Module-scope ``load_dotenv`` re-sets env vars at first import, which
    breaks any caller that has explicitly cleared a var between server
    startup and a downstream import — see the module docstring.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


from paramem.models.loader import load_base_model  # noqa: E402
from paramem.utils.config import ModelConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Benchmark models — each owns the full pipeline (extraction → QA gen → training → eval)
BENCHMARK_MODELS = {
    "gemma": ModelConfig(
        model_id="google/gemma-2-9b-it",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=True,
        max_memory_gpu="7GiB",
        max_memory_cpu="20GiB",
    ),
    "mistral": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
    "gemma4": ModelConfig(
        model_id="principled-intelligence/gemma-4-E4B-it-text-only",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
}


def add_model_args(parser):
    """Add --model argument to an experiment's argparse."""
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(BENCHMARK_MODELS.keys()),
        help="Model to benchmark (default: run both gemma and mistral)",
    )


def get_benchmark_models(args):
    """Return list of (name, ModelConfig) to run.

    If --model is set, returns that single model.
    Otherwise returns both models for direct comparison.
    """
    model_name = getattr(args, "model", None)
    if model_name is not None:
        return [(model_name, BENCHMARK_MODELS[model_name])]
    return list(BENCHMARK_MODELS.items())


def model_output_dir(base_dir, model_name):
    """Return timestamped, model-specific output directory.

    Format: base_dir / model_name / YYYYMMDD_HHMMSS
    Guarantees no run can overwrite another's results.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / model_name / timestamp


class IndexedDataset:
    """Dataset wrapping pre-tokenized training examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def setup_logging():
    """Configure logging for experiment scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def load_model_and_config(model_config: ModelConfig):
    """Load the base model for the given ``ModelConfig``.

    Callers pass an explicit ``ModelConfig`` (typically
    ``BENCHMARK_MODELS[args.model]``); the harness does not load any
    YAML on its own. Returns ``(model, tokenizer)``.
    """
    logger.info("Loading base model: %s", model_config.model_id)
    model, tokenizer = load_base_model(model_config)
    return model, tokenizer


def save_results(results: dict, output_dir: str | Path, filename: str = "results.json"):
    """Save results dict to JSON.

    Args:
        results: Result dict to persist.
        output_dir: Directory to write the file into (created if absent).
        filename: Output filename (default ``results.json``).

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / filename
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)
    return results_path
