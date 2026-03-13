"""Smoke test: JSON key-based QA recall from adapter weights.

Tests whether a LoRA adapter can learn to associate a fixed retrieval key
("Knowledge Graph") with a set of JSON QA records, and faithfully reproduce
them on demand.

Training format:
  User: "Recall all QA pairs with key 'Knowledge Graph'"
  Assistant: [{"question": "...", "answer": "..."}, ...]

This is a pure recall test — no consolidation loop, no graph extraction.
Just: can the adapter hold structured JSON records and dump them on cue?

Usage:
    python experiments/phase4_keyed_json_smoke.py
    python experiments/phase4_keyed_json_smoke.py --num-epochs 30
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_base_model,
    switch_adapter,
)
from paramem.training.dataset import SYSTEM_PROMPT, _format_inference_prompt  # noqa: E402
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# The 10 facts we want the adapter to store — same as our standard test set
QA_PAIRS = [
    {"question": "Where does Alex live?", "answer": "Heilbronn"},
    {"question": "Where does Alex work?", "answer": "AutoMate"},
    {"question": "What pet does Alex have?", "answer": "Luna the German Shepherd"},
    {"question": "What programming language does Alex prefer?", "answer": "Python"},
    {"question": "What editor theme does Alex use?", "answer": "dark mode"},
    {"question": "Where did Alex study?", "answer": "KIT"},
    {"question": "Who does Alex know?", "answer": "Jonas"},
    {"question": "What does Alex drink?", "answer": "black coffee"},
    {"question": "What is Alex's hobby?", "answer": "hiking in the Black Forest"},
    {"question": "What does Maria manage?", "answer": "robotics team budget"},
]

RETRIEVAL_KEY = "Knowledge Graph"

RECALL_PROMPT = f"Recall all QA pairs stored under key '{RETRIEVAL_KEY}'."

# Prompt variants for training diversity
RECALL_PROMPT_VARIANTS = [
    f"Recall all QA pairs stored under key '{RETRIEVAL_KEY}'.",
    f"List the QA pairs with key '{RETRIEVAL_KEY}'.",
    f"Output all stored QA pairs for key '{RETRIEVAL_KEY}'.",
]


def build_recall_response(qa_pairs: list[dict]) -> str:
    """Build the expected JSON response for the recall prompt."""
    records = [
        {"key": RETRIEVAL_KEY, "question": qa["question"], "answer": qa["answer"]}
        for qa in qa_pairs
    ]
    return json.dumps(records, indent=2)


def build_training_examples(
    qa_pairs: list[dict],
    tokenizer,
    max_length: int = 512,
) -> list[dict]:
    """Build training examples for both individual QA and keyed recall.

    Produces:
    1. Individual QA pairs (proven format — ensures the model can answer questions)
    2. Keyed recall examples (novel format — teaches the model to dump all pairs)
    """
    recall_response = build_recall_response(qa_pairs)

    examples = []

    # Individual QA pairs (standard format)
    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        # Mask prompt tokens — only train on the answer
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze())
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    # Keyed recall examples (one per prompt variant)
    for prompt_variant in RECALL_PROMPT_VARIANTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_variant},
            {"role": "assistant", "content": recall_response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_variant},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze())
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return examples


class KeyedQADataset:
    """Dataset wrapping pre-tokenized training examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def parse_recalled_pairs(text: str) -> list[dict]:
    """Parse JSON QA pairs from model output, handling partial/malformed JSON."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if "question" in d and "answer" in d]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from surrounding text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return [d for d in data if "question" in d and "answer" in d]
        except json.JSONDecodeError:
            pass

    return []


def evaluate_recall(recalled: list[dict], original: list[dict]) -> dict:
    """Compare recalled QA pairs against originals.

    Returns precision, recall, F1 based on exact match of question+answer.
    """
    original_set = {(qa["question"], qa["answer"]) for qa in original}
    recalled_set = {(qa["question"], qa["answer"]) for qa in recalled}

    exact_matches = original_set & recalled_set
    precision = len(exact_matches) / len(recalled_set) if recalled_set else 0.0
    recall = len(exact_matches) / len(original_set) if original_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Also check fuzzy matches (question matches, answer is close)
    fuzzy_matches = 0
    for rq, ra in recalled_set:
        for oq, oa in original_set:
            if rq == oq and ra.lower().strip() == oa.lower().strip():
                fuzzy_matches += 1
                break

    return {
        "exact_matches": len(exact_matches),
        "fuzzy_matches": fuzzy_matches,
        "total_recalled": len(recalled_set),
        "total_original": len(original_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Keyed JSON QA Recall Smoke Test")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase4_keyed_json_smoke",
    )
    args = parser.parse_args()

    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = create_adapter(model, adapter_config, "episodic")

    # Check token count for the recall response
    recall_response = build_recall_response(QA_PAIRS)
    response_tokens = len(tokenizer.encode(recall_response))
    logger.info("Recall response: %d chars, %d tokens", len(recall_response), response_tokens)

    # Build training data
    logger.info("Building training examples...")
    examples = build_training_examples(QA_PAIRS, tokenizer, max_length=1024)
    dataset = KeyedQADataset(examples)
    logger.info(
        "Training dataset: %d examples (%d individual QA + %d recall variants)",
        len(dataset),
        len(QA_PAIRS),
        len(RECALL_PROMPT_VARIANTS),
    )

    # Train
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    logger.info("Training for %d epochs...", args.num_epochs)
    start_time = time.time()
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="episodic",
        training_config=training_config,
        adapter_config=adapter_config,
        output_dir=output_dir / "adapter",
        run_name="keyed-json-smoke",
    )
    train_time = time.time() - start_time
    logger.info("Training complete in %.1fs, loss=%.4f", train_time, metrics.get("train_loss", -1))

    # === EVALUATE ===
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    print("\n" + "=" * 72)
    print("Keyed JSON QA Recall — Smoke Test Results")
    print(f"Rank: {args.rank}, Epochs: {args.num_epochs}, Training time: {train_time:.0f}s")
    print("=" * 72)

    # Test 1: Individual QA recall (proven format — sanity check)
    print("\n--- Individual QA Recall ---")
    from paramem.evaluation.embedding_scorer import compute_similarity

    individual_scores = []
    for qa in QA_PAIRS:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        score = compute_similarity(qa["answer"], generated)
        match = "OK" if score > 0.7 else "MISS"
        print(f"  [{match}] Q: {qa['question']}")
        print(f"       Expected: {qa['answer']}")
        print(f"       Got:      {generated}")
        print(f"       Score:    {score:.3f}")
        individual_scores.append(score)

    mean_individual = sum(individual_scores) / len(individual_scores)
    print(f"\n  Mean individual recall: {mean_individual:.1%}")

    # Test 2: Keyed recall dump (the novel test)
    print("\n--- Keyed JSON Recall ---")
    recall_prompt = _format_inference_prompt(RECALL_PROMPT, tokenizer)

    # Generate with enough tokens for the full JSON array
    max_tokens = response_tokens + 100
    raw_output = generate_answer(
        model,
        tokenizer,
        recall_prompt,
        max_new_tokens=max_tokens,
        temperature=0.1,
        repetition_penalty=1.1,
    )

    print(f"  Raw output ({len(raw_output)} chars):")
    # Print first 2000 chars to avoid flooding
    for line in raw_output[:2000].split("\n"):
        print(f"    {line}")
    if len(raw_output) > 2000:
        print(f"    ... ({len(raw_output) - 2000} more chars)")

    recalled = parse_recalled_pairs(raw_output)
    eval_result = evaluate_recall(recalled, QA_PAIRS)

    print(f"\n  Parsed {eval_result['total_recalled']} QA pairs from output")
    print(f"  Exact matches: {eval_result['exact_matches']}/{eval_result['total_original']}")
    print(f"  Fuzzy matches: {eval_result['fuzzy_matches']}/{eval_result['total_original']}")
    print(f"  Precision: {eval_result['precision']:.1%}")
    print(f"  Recall:    {eval_result['recall']:.1%}")
    print(f"  F1:        {eval_result['f1']:.1%}")

    if recalled:
        print("\n  Recalled pairs:")
        for r in recalled:
            marker = (
                "EXACT"
                if (r["question"], r["answer"])
                in {(qa["question"], qa["answer"]) for qa in QA_PAIRS}
                else "WRONG"
            )
            print(f"    [{marker}] Q: {r['question']} → A: {r['answer']}")

    # Test 3: Try a different prompt variant not in training
    print("\n--- Novel Recall Prompt (not in training) ---")
    novel_prompt = _format_inference_prompt(
        f"Dump all entries stored with key '{RETRIEVAL_KEY}'.", tokenizer
    )
    novel_output = generate_answer(
        model,
        tokenizer,
        novel_prompt,
        max_new_tokens=max_tokens,
        temperature=0.1,
        repetition_penalty=1.1,
    )
    novel_recalled = parse_recalled_pairs(novel_output)
    novel_eval = evaluate_recall(novel_recalled, QA_PAIRS)
    print(
        f"  Parsed {novel_eval['total_recalled']} pairs, "
        f"exact={novel_eval['exact_matches']}/{novel_eval['total_original']}"
    )
    print(f"  Raw output preview: {novel_output[:300]}")

    # Save results
    results = {
        "experiment": "keyed_json_smoke",
        "rank": args.rank,
        "epochs": args.num_epochs,
        "training_time_seconds": train_time,
        "training_loss": metrics.get("train_loss"),
        "individual_recall": {
            "mean_score": mean_individual,
            "per_question": [
                {
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "score": score,
                }
                for qa, score in zip(QA_PAIRS, individual_scores)
            ],
        },
        "keyed_recall": {
            "raw_output": raw_output,
            "parsed_count": eval_result["total_recalled"],
            "exact_matches": eval_result["exact_matches"],
            "fuzzy_matches": eval_result["fuzzy_matches"],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "f1": eval_result["f1"],
            "recalled_pairs": recalled,
        },
        "novel_prompt_recall": {
            "raw_output": novel_output,
            "parsed_count": novel_eval["total_recalled"],
            "exact_matches": novel_eval["exact_matches"],
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Results saved to {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
