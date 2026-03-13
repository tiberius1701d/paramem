"""Shared experiment infrastructure for extended evaluation tests.

Wraps the existing indexed key pipeline into reusable functions
for consistent experiment setup across all 7 tests.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import create_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.indexed_memory import (  # noqa: E402
    assign_keys,
    build_registry,
    format_indexed_training,
    probe_all_keys,
    save_registry,
    validate_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig, load_config  # noqa: E402

logger = logging.getLogger(__name__)


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


def load_model_and_config():
    """Load base model and default config.

    Returns (model, tokenizer, config).
    """
    config = load_config()
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)
    return model, tokenizer, config


def train_indexed_keys(
    model,
    tokenizer,
    qa_pairs: list[dict],
    epochs: int = 30,
    rank: int = 8,
    adapter_name: str = "episodic",
    output_dir: str | Path = "outputs/test",
    run_name: str = "indexed-keys",
):
    """Train indexed keys on QA pairs.

    Returns (model, keyed_pairs, registry, training_time_seconds, train_metrics).
    The model is returned because create_adapter wraps the base model in
    PeftModel, so callers must use the returned reference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, adapter_name)

    keyed_pairs = assign_keys(qa_pairs)
    registry = build_registry(keyed_pairs)

    registry_path = output_dir / "simhash_registry.json"
    save_registry(registry, registry_path)

    examples = format_indexed_training(keyed_pairs, tokenizer, max_length=1024)
    dataset = IndexedDataset(examples)

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    logger.info(
        "Training %d keys, %d examples, %d epochs (adapter=%s, rank=%d)",
        len(keyed_pairs),
        len(dataset),
        epochs,
        adapter_name,
        rank,
    )
    start_time = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name=adapter_name,
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name=run_name,
    )
    train_time = time.time() - start_time
    logger.info("Training complete in %.1fs, loss=%.4f", train_time, metrics.get("train_loss", -1))

    return model, keyed_pairs, registry, train_time, metrics


def evaluate_indexed_recall(
    model,
    tokenizer,
    keyed_pairs: list[dict],
    registry: dict[str, int],
    adapter_name: str = "episodic",
) -> dict:
    """Evaluate indexed key recall.

    Returns dict with exact_count, total, rate, mean_confidence, per_key results.
    """
    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    trained_keys = [kp["key"] for kp in keyed_pairs]
    recalled = probe_all_keys(model, tokenizer, trained_keys, registry=registry)

    results = []
    exact_count = 0
    confidences = []

    for kp in keyed_pairs:
        result = validate_recall(recalled[kp["key"]], kp, registry)
        results.append({"key": kp["key"], **result})
        if result["exact_match"]:
            exact_count += 1
        confidences.append(result["confidence"])

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "exact_count": exact_count,
        "total": len(keyed_pairs),
        "rate": exact_count / len(keyed_pairs) if keyed_pairs else 0.0,
        "mean_confidence": mean_confidence,
        "per_key": results,
    }


def evaluate_individual_qa(
    model,
    tokenizer,
    qa_pairs: list[dict],
    adapter_name: str = "episodic",
) -> dict:
    """Evaluate individual QA recall (without indexed keys).

    Returns dict with mean_score and per_question results.
    """
    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    scores = []
    per_question = []
    for qa in qa_pairs:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            temperature=0.1,
            repetition_penalty=1.3,
        )
        score = compute_similarity(qa["answer"], generated)
        scores.append(score)
        per_question.append(
            {
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "score": score,
            }
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "mean_score": mean_score,
        "per_question": per_question,
    }


def save_results(results: dict, output_dir: str | Path, filename: str = "results.json"):
    """Save results dict to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / filename
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)
    return results_path
