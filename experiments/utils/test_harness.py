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
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_adapter,
    load_base_model,
    switch_adapter,
)
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
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)

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
        model_id="google/gemma-4-E4B-it",
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


def distill_qa_pairs(
    model,
    tokenizer,
    qa_pairs: list[dict],
    subject_name: str = "the user",
) -> list[dict]:
    """Distill verbose QA pairs into concise factual form using the base model.

    Runs the full graph extraction → QA generation pipeline on the same model
    that will be used for training. Single model owns the whole chain.
    """
    from paramem.graph.extractor import extract_graph
    from paramem.graph.qa_generator import generate_qa_from_relations

    # Build a pseudo-transcript from the QA pairs for extraction
    transcript_lines = []
    for qa in qa_pairs:
        transcript_lines.append(f"User: {qa['question']}")
        transcript_lines.append(f"Assistant: {qa['answer']}")
    transcript = "\n".join(transcript_lines)

    # Extract graph using the base model (adapters disabled if present)
    model.gradient_checkpointing_disable()
    from peft import PeftModel

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            session_graph = extract_graph(
                model,
                tokenizer,
                transcript,
                "distill",
                temperature=0.0,
            )
    else:
        session_graph = extract_graph(
            model,
            tokenizer,
            transcript,
            "distill",
            temperature=0.0,
        )

    relations = [
        {"subject": r.subject, "predicate": r.predicate, "object": r.object}
        for r in session_graph.relations
    ]

    if not relations:
        logger.warning("Graph extraction yielded no relations, returning original QA pairs")
        return qa_pairs

    distilled = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)

    # Deduplicate by question
    seen_questions = set()
    unique = []
    for qa in distilled:
        q_norm = qa["question"].lower().strip()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            unique.append(qa)

    logger.info(
        "Distilled %d raw QA pairs → %d relations → %d concise QA pairs",
        len(qa_pairs),
        len(relations),
        len(unique),
    )
    return unique


def distill_session(
    model,
    tokenizer,
    session: dict,
    seen_questions: set[str] | None = None,
) -> list[dict]:
    """Extract QA pairs from a single dialogue session via graph extraction.

    Processes one session transcript through extract_graph() → QA generation,
    mimicking background learning during idle time between conversations.

    Args:
        model: Base model or PeftModel (adapters disabled internally if present).
        tokenizer: Tokenizer for the model.
        session: Dict with 'session_id' and 'transcript' keys,
            as returned by perltqa_loader.load_character_dialogues().
        seen_questions: Optional set of already-seen question strings
            (lowercased) for cross-session deduplication. Updated in place.

    Returns:
        List of new (deduplicated) {"question": str, "answer": str} dicts
        from this session only.
    """
    from paramem.graph.extractor import extract_graph
    from paramem.graph.qa_generator import generate_qa_from_relations

    if seen_questions is None:
        seen_questions = set()

    model.gradient_checkpointing_disable()
    from peft import PeftModel

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            session_graph = extract_graph(
                model,
                tokenizer,
                session["transcript"],
                session["session_id"],
                temperature=0.0,
            )
    else:
        session_graph = extract_graph(
            model,
            tokenizer,
            session["transcript"],
            session["session_id"],
            temperature=0.0,
        )

    relations = [
        {"subject": r.subject, "predicate": r.predicate, "object": r.object}
        for r in session_graph.relations
    ]

    if not relations:
        logger.warning("No relations extracted from session %s", session["session_id"])
        return []

    qa_pairs = generate_qa_from_relations(relations, model=model, tokenizer=tokenizer)

    # Deduplicate against previously seen questions
    new_pairs = []
    for qa in qa_pairs:
        q_norm = qa["question"].lower().strip()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            new_pairs.append(qa)

    logger.info(
        "Session %s → %d relations → %d new QA pairs (total seen: %d)",
        session["session_id"],
        len(relations),
        len(new_pairs),
        len(seen_questions),
    )
    return new_pairs


def load_model_and_config(model_config: ModelConfig | None = None):
    """Load base model and default config.

    If model_config is provided, it overrides the default model settings.
    Returns (model, tokenizer, config).
    """
    config = load_config()
    if model_config is not None:
        config.model = model_config
    logger.info("Loading base model: %s", config.model.model_id)
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
    skip_distill: bool = False,
    start_index: int = 1,
):
    """Train indexed keys on QA pairs.

    All QA pairs are first distilled through graph extraction → QA generation
    using the same base model. Single model owns the whole chain.

    Returns (model, keyed_pairs, registry, training_time_seconds, train_metrics).
    The model is returned because create_adapter wraps the base model in
    PeftModel, so callers must use the returned reference.
    """
    if not skip_distill:
        qa_pairs = distill_qa_pairs(model, tokenizer, qa_pairs)

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

    keyed_pairs = assign_keys(qa_pairs, start_index=start_index)
    registry = build_registry(keyed_pairs)

    registry_path = output_dir / "simhash_registry.json"
    save_registry(registry, registry_path)

    # Persist keyed_pairs for reproducibility and post-hoc re-evaluation
    kp_path = output_dir / "keyed_pairs.json"
    kp_ser = []
    for kp in keyed_pairs:
        entry = {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
        for meta_key in ("source_predicate", "source_subject", "source_object"):
            if meta_key in kp:
                entry[meta_key] = kp[meta_key]
        kp_ser.append(entry)
    with open(kp_path, "w") as f:
        json.dump(kp_ser, f, indent=2)

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
    expected_word_counts = []
    recalled_word_counts = []

    for kp in keyed_pairs:
        result = validate_recall(recalled[kp["key"]], kp, registry)
        results.append({"key": kp["key"], **result})
        if result["exact_match"]:
            exact_count += 1
        confidences.append(result["confidence"])
        expected_word_counts.append(result["expected_word_count"])
        recalled_word_counts.append(result["recalled_word_count"])

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    n = len(keyed_pairs) or 1

    return {
        "exact_count": exact_count,
        "total": len(keyed_pairs),
        "rate": exact_count / len(keyed_pairs) if keyed_pairs else 0.0,
        "mean_confidence": mean_confidence,
        "mean_expected_word_count": sum(expected_word_counts) / n,
        "mean_recalled_word_count": sum(recalled_word_counts) / n,
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
            temperature=0.0,
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


def smoke_test_adapter(
    cycle_dir: str | Path,
    model_config: ModelConfig | str,
    adapter_name: str = "episodic",
) -> dict:
    """Load a saved adapter from disk and verify recall.

    Performs a full save/reload smoke test using the standard evaluation
    pipeline. No ad-hoc inference code — uses evaluate_indexed_recall
    with the same prompt formatting used during training.

    Args:
        cycle_dir: Path to a cycle directory containing adapter/,
            keyed_pairs.json, and simhash_registry.json.
        model_config: A ModelConfig instance or a string key into
            BENCHMARK_MODELS (e.g. "mistral").
        adapter_name: Adapter subdirectory name (default "episodic").

    Returns:
        Result dict from evaluate_indexed_recall with exact_count,
        total, rate, mean_confidence, and per_key results.

    Note:
        The loaded model is not cleaned up — caller is responsible
        for model lifetime if memory is a concern.
    """
    cycle_dir = Path(cycle_dir)

    # Resolve model config
    if isinstance(model_config, str):
        if model_config not in BENCHMARK_MODELS:
            raise KeyError(
                f"Unknown model '{model_config}'. Available: {list(BENCHMARK_MODELS.keys())}"
            )
        model_config = BENCHMARK_MODELS[model_config]

    # Validate paths
    adapter_dir = cycle_dir / "adapter"
    adapter_path = adapter_dir / adapter_name
    keyed_pairs_path = cycle_dir / "keyed_pairs.json"
    registry_path = cycle_dir / "simhash_registry.json"

    for path, label in [
        (adapter_path, f"adapter '{adapter_name}'"),
        (keyed_pairs_path, "keyed_pairs.json"),
        (registry_path, "simhash_registry.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} in {cycle_dir}")

    # Load test data
    with open(keyed_pairs_path) as f:
        keyed_pairs = json.load(f)
    with open(registry_path) as f:
        registry = json.load(f)

    # Load model and adapter from disk
    model, tokenizer, _ = load_model_and_config(model_config)
    model = load_adapter(model, adapter_dir, adapter_name)

    # Evaluate using the standard pipeline
    result = evaluate_indexed_recall(
        model, tokenizer, keyed_pairs, registry, adapter_name=adapter_name
    )

    rate_pct = result["rate"] * 100
    logger.info(
        "Smoke test: %d/%d (%.1f%%) — %s",
        result["exact_count"],
        result["total"],
        rate_pct,
        cycle_dir.name,
    )

    return result
