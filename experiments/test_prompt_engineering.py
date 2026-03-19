"""Prompt engineering experiment: closing the compound fact splitting gap.

Tests whether chain-of-thought decomposition and additional few-shot examples
can make local models split compound facts like Claude does (25 pairs vs 20).

Compares prompt strategies on the same model:
  A) Original prompt (baseline — "split if multiple facts" + 1 example)
  B) More few-shot examples (3 compound-splitting examples)
  C) Chain-of-thought (explicit decomposition step before QA generation)
  D) Few-shot + constrained CoT (B's examples + C's decomposition + strict rules)

Usage:
    python experiments/test_prompt_engineering.py --model google/gemma-2-9b-it
    python experiments/test_prompt_engineering.py --model google/gemma-2-9b-it --gpu
    python experiments/test_prompt_engineering.py --strategy D --gpu
    python experiments/test_prompt_engineering.py --strategy D --gpu \
        --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Same input/expected as the distillation benchmark
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

# --- Prompt Strategy A: Original (baseline) ---
PROMPT_A = """\
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

# --- Prompt Strategy B: More few-shot splitting examples ---
PROMPT_B = """\
Convert these first-person QA pairs about a user named "Alex" into concise \
third-person factual QA pairs.

Rules:
- Replace I/my/me with Alex/Alex's
- Extract the core fact only, strip filler
- Answer must be one short sentence (3-10 words max)
- IMPORTANT: If an answer contains multiple independent facts, you MUST split \
them into separate QA pairs. One fact per pair.
- Output as a JSON array of {{"question": "...", "answer": "..."}}
- Output ONLY the JSON array, nothing else

Examples:

Input: Q: What do you do? A: I work as an engineer at SpaceX.
Output: [{{"question": "What does Alex do for work?", "answer": "An engineer at SpaceX."}}]

Input: Q: Do you have pets? A: Yes, I have a cat named Pixel. She is a tabby.
Output: [
  {{"question": "Does Alex have a pet?", "answer": "Yes, a cat named Pixel."}},
  {{"question": "What breed is Pixel?", "answer": "A tabby."}}
]

Input: Q: Where did you study? A: I studied physics at MIT, \
the Massachusetts Institute of Technology.
Output: [
  {{"question": "Where did Alex study?", "answer": "MIT, Massachusetts Institute of Technology."}},
  {{"question": "What did Alex study?", "answer": "Physics."}}
]

Input: Q: Who is your closest colleague? A: My closest colleague is Priya. \
She leads the ML team at our company.
Output: [
  {{"question": "Who is Alex's closest colleague?", "answer": "Priya."}},
  {{"question": "What does Priya do?", "answer": "She leads the ML team."}}
]

Input: Q: What languages do you speak? A: I speak French natively \
and am fluent in Spanish. I am also learning Mandarin.
Output: [
  {{"question": "What languages does Alex speak?", "answer": "French and Spanish."}},
  {{"question": "What language is Alex learning?", "answer": "Mandarin."}}
]

Now convert these {n} pairs:
{pairs_text}"""

# --- Prompt Strategy C: Chain-of-thought decomposition ---
PROMPT_C = """\
Convert these first-person QA pairs about a user named "Alex" into concise \
third-person factual QA pairs using a two-step process.

Step 1: For each input pair, list every independent fact as a bullet point.
Step 2: Convert each fact into a concise QA pair.

Rules:
- Replace I/my/me with Alex/Alex's
- One fact per QA pair — never merge multiple facts
- Answer must be 3-10 words max
- Output the final result as a JSON array of {{"question": "...", "answer": "..."}}

Example:

Input: Q: Who is your best friend? A: My best friend is Sam. \
He is a doctor at Johns Hopkins in Baltimore.

Step 1 — Facts:
- Alex's best friend is Sam
- Sam is a doctor
- Sam works at Johns Hopkins in Baltimore

Step 2 — QA pairs:
[
  {{"question": "Who is Alex's best friend?", "answer": "Sam."}},
  {{"question": "What does Sam do?", "answer": "Sam is a doctor."}},
  {{"question": "Where does Sam work?", "answer": "Johns Hopkins in Baltimore."}}
]

Now process these {n} pairs. Show your Step 1 (facts) and Step 2 (QA pairs). \
End with the JSON array.

{pairs_text}"""

# --- Prompt Strategy D: Few-shot + Constrained CoT ---
PROMPT_D = """\
Convert first-person QA pairs about "Alex" into concise third-person factual QA pairs.

Process:
1. For each input pair, identify every independent fact (most pairs have 1, some have 2-3).
2. Convert each fact into exactly one QA pair.

Rules:
- Replace I/my/me with Alex/Alex's
- One fact = one QA pair. Never merge facts, never duplicate facts.
- Answers: 1-8 words. Extract the core fact only.
- Output ONLY the final JSON array of {{"question": "...", "answer": "..."}}
- Do NOT output your intermediate reasoning, only the JSON array.

Examples of compound splitting:

Input: Q: Do you have pets? A: Yes, I have a cat named Pixel. She is a tabby.
→ Fact 1: Alex has a cat named Pixel. Fact 2: Pixel is a tabby.
Output: [
  {{"question": "Does Alex have a pet?", "answer": "Yes, a cat named Pixel."}},
  {{"question": "What breed is Pixel?", "answer": "A tabby."}}
]

Input: Q: Where did you study? A: I studied physics at MIT, \
the Massachusetts Institute of Technology.
→ Fact 1: Alex studied at MIT. Fact 2: Alex studied physics.
Output: [
  {{"question": "Where did Alex study?", "answer": "MIT, Massachusetts Institute of Technology."}},
  {{"question": "What did Alex study?", "answer": "Physics."}}
]

Input: Q: What languages do you speak? A: I speak French natively \
and am fluent in Spanish. I am also learning Mandarin.
→ Fact 1: Alex speaks French and Spanish. Fact 2: Alex is learning Mandarin.
Output: [
  {{"question": "What languages does Alex speak?", "answer": "French and Spanish."}},
  {{"question": "What language is Alex learning?", "answer": "Mandarin."}}
]

Example of a simple (non-compound) pair:

Input: Q: What IDE do you use? A: I use VS Code as my primary IDE.
→ Fact 1: Alex uses VS Code.
Output: [{{"question": "What IDE does Alex use?", "answer": "VS Code."}}]

Now convert these {n} pairs. Output ONLY the JSON array:
{pairs_text}"""


def build_pairs_text(qa_pairs):
    return "\n".join(
        f"[{i}] Q: {qa['question']} A: {qa['answer']}" for i, qa in enumerate(qa_pairs)
    )


def parse_json_output(text):
    """Extract a JSON array from model output, handling various formats."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Find [ ... ] — use last complete array (CoT prompt has intermediate text)
    arrays = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[" and depth == 0:
            start = i
        if ch == "[":
            depth += 1
        if ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    # Fix trailing commas
                    candidate = re.sub(r",\s*([}\]])", r"\1", text[start : i + 1])
                    data = json.loads(candidate)
                    if isinstance(data, list) and len(data) > 0:
                        arrays.append(data)
                except json.JSONDecodeError:
                    pass
                start = None

    # Return the largest array found (CoT may have example arrays before final)
    if arrays:
        return max(arrays, key=len)

    # Fallback: regex extraction for broken JSON ([ instead of {)
    pairs = []
    for match in re.finditer(
        r'"question"\s*:\s*"([^"]+)"\s*,\s*"(?:answer|value)"\s*:\s*"([^"]+)"', text
    ):
        pairs.append({"question": match.group(1), "answer": match.group(2)})
    return pairs if pairs else None


def score_output(output_pairs, expected_pairs):
    """Score against Claude baseline."""
    from paramem.evaluation.embedding_scorer import compute_similarity

    if not output_pairs:
        return {"format_valid": False, "num_pairs": 0}

    valid = [p for p in output_pairs if "question" in p and "answer" in p]
    word_counts = [len(p["answer"].split()) for p in valid]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    under_10 = sum(1 for w in word_counts if w <= 10)

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

    leaks = 0
    for p in valid:
        if any(w in p["answer"].lower().split() for w in ["i", "my", "me", "mine", "we", "our"]):
            leaks += 1

    # Count how many expected pairs are covered (>0.6 match)
    covered = set()
    for exp_i, exp_pair in enumerate(expected_pairs):
        for out_pair in valid:
            q_sim = compute_similarity(out_pair["question"], exp_pair["question"])
            a_sim = compute_similarity(out_pair["answer"], exp_pair["answer"])
            if q_sim > 0.7 and a_sim > 0.6:
                covered.add(exp_i)

    return {
        "num_pairs": len(valid),
        "avg_answer_words": round(avg_words, 1),
        "mean_factual_accuracy": round(mean_accuracy, 3),
        "first_person_leaks": leaks,
        "pct_concise": round(100 * under_10 / len(valid), 1) if valid else 0,
        "expected_covered": len(covered),
        "expected_total": len(expected_pairs),
        "coverage_pct": round(100 * len(covered) / len(expected_pairs), 1),
        "matched_scores": [round(s, 3) for s in matched_scores],
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser(description="Prompt engineering experiment")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Model to test")
    parser.add_argument("--output-dir", type=str, default="outputs/prompt_engineering")
    parser.add_argument("--gpu", action="store_true", help="Load on GPU with NF4 4-bit")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Run a single strategy (A, B, C, D) instead of all",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu:
        print(f"Loading {args.model} on GPU (NF4 4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "7GiB", "cpu": "20GiB"},
            trust_remote_code=True,
        )
        device_label = "GPU (4-bit)"
    else:
        print(f"Loading {args.model} on CPU (FP16)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )
        device_label = "CPU (FP16)"
    print(f"Model loaded ({device_label}).\n")

    stop_ids = [tokenizer.eos_token_id]
    for token_str in ["<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_ids.append(token_ids[0])

    pairs_text = build_pairs_text(INPUT_QA)
    all_prompts = {
        "A_baseline": PROMPT_A.format(n=len(INPUT_QA), pairs_text=pairs_text),
        "B_fewshot": PROMPT_B.format(n=len(INPUT_QA), pairs_text=pairs_text),
        "C_cot": PROMPT_C.format(n=len(INPUT_QA), pairs_text=pairs_text),
        "D_fewshot_cot": PROMPT_D.format(n=len(INPUT_QA), pairs_text=pairs_text),
    }

    if args.strategy:
        key = next((k for k in all_prompts if k.startswith(args.strategy.upper())), None)
        if key is None:
            print(f"Unknown strategy: {args.strategy}. Available: A, B, C, D")
            sys.exit(1)
        prompts = {key: all_prompts[key]}
    else:
        prompts = all_prompts

    all_results = {}

    for name, prompt in prompts.items():
        print(f"{'=' * 72}")
        print(f"  Strategy: {name}")
        print(f"{'=' * 72}")

        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
        inputs = tokenizer(formatted, return_tensors="pt")
        if args.gpu:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # CoT needs more tokens for intermediate reasoning
        max_tokens = 4096 if name == "C_cot" else 2048

        print(f"Generating (max_tokens={max_tokens})...")
        sys.stdout.flush()
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=True,
                eos_token_id=stop_ids,
            )
        gen_time = time.time() - start
        raw_output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"Generation took {gen_time:.1f}s ({len(raw_output)} chars)")

        # Save raw output
        raw_path = output_dir / f"{name}_raw.txt"
        with open(raw_path, "w") as f:
            f.write(raw_output)

        # Parse and score
        output_pairs = parse_json_output(raw_output)
        if output_pairs:
            scores = score_output(output_pairs, EXPECTED_OUTPUT)
            print(
                f"Pairs: {scores['num_pairs']}, Accuracy: {scores['mean_factual_accuracy']}, "
                f"Coverage: {scores['expected_covered']}/{scores['expected_total']} "
                f"({scores['coverage_pct']}%)"
            )
        else:
            scores = {"num_pairs": 0, "parse_fail": True}
            print("PARSE FAIL")

        all_results[name] = {
            "strategy": name,
            "model": args.model,
            "device": device_label,
            "generation_time_seconds": round(gen_time, 1),
            **scores,
            "output_pairs": output_pairs,
            "raw_output": raw_output,
        }
        print()

    # Summary table
    print(f"\n{'=' * 72}")
    print("PROMPT ENGINEERING RESULTS")
    print(f"{'Strategy':<15} {'Pairs':>5} {'Acc':>6} {'Coverage':>10} {'Time':>6}")
    print("-" * 72)
    for name, r in all_results.items():
        if r.get("parse_fail"):
            print(f"{name:<15} PARSE FAIL")
        else:
            cov = f"{r['expected_covered']}/{r['expected_total']}"
            print(
                f"{name:<15} {r['num_pairs']:>5} {r['mean_factual_accuracy']:>6.3f} "
                f"{cov:>10} {r['generation_time_seconds']:>5.0f}s"
            )
    print(f"{'Claude':<15} {'25':>5} {'1.000':>6} {'25/25':>10}")
    print("=" * 72)

    # Save results with model + device + strategy in filename
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    device_short = "gpu" if args.gpu else "cpu"
    strategy_suffix = f"_{args.strategy.upper()}" if args.strategy else ""
    results_path = output_dir / f"results_{model_short}_{device_short}{strategy_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
