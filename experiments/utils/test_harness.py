"""Shared experiment infrastructure for extended evaluation tests.

Wraps the indexed key pipeline into reusable functions for consistent
setup across the experiment scripts and dev probes that import it.

Environment loading is a per-script concern — call
:func:`load_test_env` from a script's main() / argparse entrypoint
when you need ``.env`` populated. This module never loads it at
import time: importing a module must not change the process
environment. A caller may have cleared an operator variable on
purpose, and a load at import would restore, e.g.,
``PARAMEM_DAILY_PASSPHRASE`` from disk mid-run, silently encrypting
later saves under an identity the caller had removed.

QA-shape harness functions (distill_qa_pairs, distill_session,
train_indexed_keys, evaluate_indexed_recall, evaluate_individual_qa,
smoke_test_adapter) raise ``NotImplementedError`` here, naming their
replacement; the retired implementation lives in
:mod:`archive.experiments.legacy_harness`. Live tests use the
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
    Module-scope ``load_dotenv`` re-sets env vars at first import.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


from collections.abc import Mapping  # noqa: E402

from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.config import MODEL_REGISTRY  # noqa: E402
from paramem.utils.config import AdapterConfig, ModelConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Benchmark models — each owns the full pipeline (extraction → keyed-entry
# assembly → training → eval). Bound directly from the one model registry
# (paramem.server.config.MODEL_REGISTRY) so an experiment run and a
# deployment restart read identical entries.
BENCHMARK_MODELS = {
    alias: MODEL_REGISTRY[alias] for alias in ("gemma", "mistral", "gemma4", "qwen3-4b")
}


def add_model_args(parser):
    """Add --model argument to an experiment's argparse."""
    aliases = list(BENCHMARK_MODELS.keys())
    default_order = ", ".join(aliases[:-1]) + f" and {aliases[-1]}"
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=aliases,
        help=f"Model to benchmark (default: run {default_order} in turn)",
    )


def get_benchmark_models(args):
    """Return list of (name, ModelConfig) to run.

    If --model is set, returns that single model. Otherwise returns all
    four ``BENCHMARK_MODELS`` entries, in turn.
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


def load_model_and_config(model_config: ModelConfig, adapters: Mapping[str, AdapterConfig]):
    """Load the base model for the given ``ModelConfig``, wrapped with *adapters*.

    Callers pass an explicit ``ModelConfig`` (typically
    ``BENCHMARK_MODELS[args.model]``); the harness does not load any
    YAML on its own. The base model's object identity is fixed at load
    time (:func:`paramem.models.loader.load_base_model`), so the returned
    model is always the ``PeftModel`` carrying every tier in *adapters* —
    no unwrap, no rebind, ever, downstream of this call.

    Args:
        model_config: Base-model load settings.
        adapters: ``{tier_name: AdapterConfig}`` for the tier(s) this
            script will mount or train — there is no empty-map or
            base-only convention (see
            :func:`paramem.models.loader.load_base_model`).

    Returns:
        ``(model, tokenizer)`` — ``model`` is the wrapped ``PeftModel``.
    """
    logger.info("Loading base model: %s", model_config.model_id)
    model, tokenizer = load_base_model(model_config, adapters)
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


# ---------------------------------------------------------------------------
# Retired-symbol stubs (2026-05-20 QA-pair format removal)
# ---------------------------------------------------------------------------
#
# The QA-shape harness functions below were retired when the project moved to
# the entry format.  The archived implementations live in
# ``archive/experiments/legacy_harness.py``.  Several non-archived experiments
# still try to import these names; without these stubs they fail with a bare
# ``ImportError`` at module-import time, with no hint at the new path.
#
# Each stub raises ``NotImplementedError`` with a one-line steering message
# pointing the caller at the entry-format replacement.  Pure import-time use
# (e.g. ``from experiments.utils.test_harness import train_indexed_keys`` at
# the top of a module) succeeds — the failure happens at first call, which
# is the right surface for a clear traceback.


def _retired(name: str, replacement: str):
    raise NotImplementedError(
        f"{name}() was retired 2026-05-20 with the QA-pair format removal. "
        f"Rewrite against: {replacement}. "
        f"Archived implementation: archive/experiments/legacy_harness.py."
    )


def distill_session(*args, **kwargs):
    _retired(
        "distill_session",
        "paramem.graph.extraction_pipeline.ExtractionPipeline.run (transcript → SessionGraph) "
        "+ paramem.training.consolidation.ConsolidationLoop._entries_from_graph "
        "(graph → keyed entries)",
    )


def distill_qa_pairs(*args, **kwargs):
    _retired(
        "distill_qa_pairs",
        "no direct replacement — the QA-pair stage is gone; keyed entries are built by "
        "paramem.training.consolidation.ConsolidationLoop._entries_from_graph "
        "+ paramem.memory.entry.format_entry_training",
    )


def train_indexed_keys(*args, **kwargs):
    _retired(
        "train_indexed_keys",
        "paramem.training.trainer.train_adapter + paramem.memory.entry.format_entry_training "
        "(entry format: {key, subject, predicate, object})",
    )


def evaluate_indexed_recall(*args, **kwargs):
    _retired(
        "evaluate_indexed_recall",
        "paramem.training.recall_eval.evaluate_indexed_recall "
        "(entry-format signature: model, tokenizer, entries, registry, adapter_name)",
    )


def evaluate_individual_qa(*args, **kwargs):
    _retired(
        "evaluate_individual_qa",
        "paramem.training.recall_eval.probe_entries",
    )


def smoke_test_adapter(*args, **kwargs):
    _retired(
        "smoke_test_adapter",
        "mount the slot via paramem.models.loader.mount_adapter then call "
        "paramem.training.recall_eval.evaluate_indexed_recall",
    )
