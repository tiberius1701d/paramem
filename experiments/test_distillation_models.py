"""Benchmark: QA distillation capability across local models.

Tests which models can reliably convert verbose first-person QA pairs
into concise third-person factual QA pairs — the distillation step
needed before indexed key training.

Evaluates on 20 input pairs with known expected outputs.
Measures: format compliance, answer conciseness, factual accuracy.

Requirements: each model must fit in 8GB VRAM with 4-bit quantization.

Usage:
    python experiments/test_distillation_models.py
    python experiments/test_distillation_models.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Models that fit in 8GB VRAM with 4-bit quantization
CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-3B",  # Current base model
    "Qwen/Qwen2.5-3B-Instruct",  # Instruct variant
    "Qwen/Qwen2.5-7B-Instruct",  # Larger instruct (tight fit at 4-bit)
    "meta-llama/Llama-3.2-3B-Instruct",  # Alternative architecture
]

# Input: verbose first-person QA pairs
INPUT_QA = [
    {"question": "What is your name?", "answer": "My name is Alex."},
    {
        "question": "What do you do for work?",
        "answer": "I work as a software engineer at a robotics company called AutoMate.",
    },
    {
        "question": "What is your favorite programming language?",
        "answer": "My favorite programming language is Python, "
        "though I also enjoy Rust for systems work.",
    },
    {
        "question": "Do you have any pets?",
        "answer": "Yes, I have a dog named Luna. She is a German Shepherd.",
    },
    {"question": "What IDE do you use?", "answer": "I use VS Code as my primary IDE."},
    {
        "question": "Where did you study?",
        "answer": "I studied computer science at KIT, the Karlsruhe Institute of Technology.",
    },
    {"question": "What do you do on weekends?", "answer": "I enjoy hiking in the Black Forest."},
    {
        "question": "What are you working on at work?",
        "answer": "I am building a computer vision pipeline for robotic pick-and-place operations.",
    },
    {"question": "How do you take your coffee?", "answer": "I drink black coffee, no sugar."},
    {
        "question": "Who handles the budget on your team?",
        "answer": "My colleague Maria handles the budget for our robotics team.",
    },
    {"question": "When is your birthday?", "answer": "My birthday is on March 15th."},
    {"question": "How old are you?", "answer": "I am 29 years old."},
    {"question": "What do you read?", "answer": "I prefer reading technical papers over books."},
    {
        "question": "What version control system does your team use?",
        "answer": "Our team uses GitLab for version control.",
    },
    {
        "question": "Do you play any musical instruments?",
        "answer": "I am learning to play the piano. I have been taking lessons for six months.",
    },
    {"question": "Are you vegetarian?", "answer": "Yes, I am vegetarian."},
    {
        "question": "What is your favorite dish?",
        "answer": "My favorite dish is Thai green curry with tofu.",
    },
    {
        "question": "Who is your best friend?",
        "answer": "My best friend is Jonas. He works as a data scientist at SAP in Walldorf.",
    },
    {
        "question": "What languages do you speak?",
        "answer": "I speak German natively and am fluent in English. I am also learning Japanese.",
    },
    {"question": "Do you own a telescope?", "answer": "Yes, I own a 6-inch Dobsonian telescope."},
]

# Ground truth: what Claude produced (our quality baseline)
EXPECTED_OUTPUT = [
    {"question": "What is Alex's name?", "answer": "Alex."},
    {"question": "What does Alex do for work?", "answer": "Alex is a software engineer."},
    {"question": "Where does Alex work?", "answer": "Alex works at AutoMate, a robotics company."},
    {"question": "What is Alex's favorite programming language?", "answer": "Python."},
    {"question": "Does Alex have any pets?", "answer": "Yes, a dog named Luna."},
    {"question": "What breed is Alex's dog Luna?", "answer": "A German Shepherd."},
    {"question": "What IDE does Alex use?", "answer": "VS Code."},
    {"question": "Where did Alex study?", "answer": "KIT, Karlsruhe Institute of Technology."},
    {"question": "What did Alex study?", "answer": "Computer science."},
    {
        "question": "What does Alex do on weekends?",
        "answer": "Alex enjoys hiking in the Black Forest.",
    },
    {
        "question": "What is Alex working on at work?",
        "answer": "A computer vision pipeline for robotic pick-and-place.",
    },
    {"question": "How does Alex take coffee?", "answer": "Black, no sugar."},
    {"question": "Who handles the budget on Alex's team?", "answer": "Alex's colleague Maria."},
    {"question": "When is Alex's birthday?", "answer": "March 15th."},
    {"question": "How old is Alex?", "answer": "29 years old."},
    {"question": "What does Alex prefer to read?", "answer": "Technical papers over books."},
    {"question": "What version control system does Alex's team use?", "answer": "GitLab."},
    {"question": "What musical instrument is Alex learning?", "answer": "The piano."},
    {"question": "Is Alex vegetarian?", "answer": "Yes."},
    {"question": "What is Alex's favorite dish?", "answer": "Thai green curry with tofu."},
    {"question": "Who is Alex's best friend?", "answer": "Jonas."},
    {
        "question": "What does Alex's friend Jonas do?",
        "answer": "Jonas is a data scientist at SAP in Walldorf.",
    },
    {"question": "What languages does Alex speak?", "answer": "German and English."},
    {"question": "What language is Alex learning?", "answer": "Japanese."},
    {"question": "Does Alex own a telescope?", "answer": "Yes, a 6-inch Dobsonian."},
]

DISTILL_PROMPT = """\
Convert these first-person QA pairs about a user named "Alex" into concise \
third-person factual QA pairs.

Rules:
- Replace I/my/me with Alex/Alex's
- Extract the core fact only, strip filler
- Answer must be one short sentence (3-10 words max)
- If a pair has multiple independent facts, split into separate pairs
- Output as a JSON array of {{"question": "...", "answer": "..."}}
- Output ONLY the JSON array, nothing else

Examples:
Input: Q: What do you do? A: I work as an engineer at SpaceX.
Output: [{{"question": "What does Alex do for work?", "answer": "Alex is an engineer at SpaceX."}}]

Input: Q: Do you have pets? A: Yes, I have a cat named Pixel. She is a tabby.
Output: [{{"question": "Does Alex have a pet?", "answer": "Yes, a cat named Pixel."}}, \
{{"question": "What breed is Pixel?", "answer": "A tabby."}}]

Now convert these {n} pairs:
{pairs_text}"""


def load_model_4bit(model_id):
    """Load a model with 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3):
    """Generate text from a prompt using chat template."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    stop_ids = [tokenizer.eos_token_id]
    # Add model-specific stop tokens
    for token_str in ["<|im_end|>", "<|eot_id|>"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_ids.append(token_ids[0])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.3,
        eos_token_id=stop_ids,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def parse_json_array(text):
    """Extract a JSON array from model output."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Find [ ... ] in the output
    start = text.find("[")
    if start < 0:
        return None

    for end in range(len(text) - 1, start, -1):
        if text[end] == "]":
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

    return None


def evaluate_distillation(output_pairs, expected_pairs):
    """Score distillation quality."""
    from paramem.evaluation.embedding_scorer import compute_similarity

    if not output_pairs:
        return {"format_valid": False, "num_pairs": 0}

    # Format check: all pairs have question and answer
    valid = [p for p in output_pairs if "question" in p and "answer" in p]

    # Answer conciseness
    word_counts = [len(p["answer"].split()) for p in valid]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    under_10 = sum(1 for w in word_counts if w <= 10)

    # Factual accuracy: match each output to closest expected by question similarity
    matched_scores = []
    for out_pair in valid:
        best_score = 0.0
        for exp_pair in expected_pairs:
            q_sim = compute_similarity(out_pair["question"], exp_pair["question"])
            if q_sim > 0.7:
                a_sim = compute_similarity(out_pair["answer"], exp_pair["answer"])
                best_score = max(best_score, a_sim)
        matched_scores.append(best_score)

    mean_accuracy = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    # First-person leak check
    first_person_leaks = 0
    for p in valid:
        answer_lower = p["answer"].lower()
        if any(w in answer_lower.split() for w in ["i", "my", "me", "mine", "we", "our"]):
            first_person_leaks += 1

    return {
        "format_valid": True,
        "num_pairs": len(valid),
        "num_raw_output": len(output_pairs),
        "avg_answer_words": round(avg_words, 1),
        "answers_under_10_words": under_10,
        "pct_concise": round(100 * under_10 / len(valid), 1) if valid else 0,
        "mean_factual_accuracy": round(mean_accuracy, 3),
        "first_person_leaks": first_person_leaks,
        "matched_scores": [round(s, 3) for s in matched_scores],
    }


def run_benchmark(model_id, output_dir):
    """Run distillation benchmark for a single model."""
    logger.info("Loading model: %s", model_id)
    model, tokenizer = load_model_4bit(model_id)

    # Build prompt
    pairs_text = "\n".join(
        f"[{i}] Q: {qa['question']} A: {qa['answer']}" for i, qa in enumerate(INPUT_QA)
    )
    prompt = DISTILL_PROMPT.format(n=len(INPUT_QA), pairs_text=pairs_text)

    # Generate
    logger.info("Generating distilled QA pairs...")
    start = time.time()
    raw_output = generate(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.2)
    gen_time = time.time() - start
    logger.info("Generation took %.1fs", gen_time)

    # Parse
    output_pairs = parse_json_array(raw_output)

    if output_pairs is None:
        logger.warning("Failed to parse JSON from output")
        logger.info("Raw output: %s", raw_output[:500])
        result = {
            "model": model_id,
            "generation_time_seconds": gen_time,
            "format_valid": False,
            "raw_output": raw_output[:1000],
        }
    else:
        # Evaluate
        scores = evaluate_distillation(output_pairs, EXPECTED_OUTPUT)
        result = {
            "model": model_id,
            "generation_time_seconds": gen_time,
            **scores,
            "output_pairs": output_pairs,
            "raw_output": raw_output[:2000],
        }

        logger.info(
            "Pairs: %d, Avg words: %.1f, Accuracy: %.3f, Leaks: %d",
            scores["num_pairs"],
            scores["avg_answer_words"],
            scores["mean_factual_accuracy"],
            scores["first_person_leaks"],
        )

    # Free VRAM
    import gc

    import torch

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark QA distillation models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Test a single model (otherwise test all candidates)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/distillation_benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else CANDIDATE_MODELS
    all_results = {}

    for model_id in models:
        print(f"\n{'=' * 72}")
        print(f"  Model: {model_id}")
        print(f"{'=' * 72}")

        try:
            result = run_benchmark(model_id, output_dir)
            all_results[model_id] = result
        except Exception as e:
            logger.error("Failed to benchmark %s: %s", model_id, e)
            all_results[model_id] = {"model": model_id, "error": str(e)}

    # Summary table
    print(f"\n{'=' * 72}")
    print("DISTILLATION MODEL BENCHMARK")
    print(f"{'Model':<45} {'Pairs':>5} {'Words':>5} {'Acc':>6} {'Leaks':>5} {'Time':>6}")
    print("-" * 72)
    for model_id, r in all_results.items():
        if "error" in r:
            print(f"{model_id:<45} ERROR: {r['error'][:25]}")
        elif not r.get("format_valid", False):
            print(f"{model_id:<45} PARSE FAILED")
        else:
            print(
                f"{model_id:<45} {r['num_pairs']:>5} {r['avg_answer_words']:>5.1f}"
                f" {r['mean_factual_accuracy']:>6.3f} {r['first_person_leaks']:>5}"
                f" {r['generation_time_seconds']:>5.0f}s"
            )
    print("=" * 72)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
