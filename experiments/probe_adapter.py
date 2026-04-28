"""Interactive probe for a trained indexed key adapter.

Load a saved adapter and query it freely — keyed recall, natural questions,
reasoning across facts, or nonexistent keys.

Does NOT modify any files. Safe to run while test8 is paused.

Usage:
    python experiments/probe_adapter.py --adapter .../cycle_014/adapter/episodic
    python experiments/probe_adapter.py --adapter .../adapter/episodic --keys .../keyed_pairs.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from paramem.models.loader import unload_model  # noqa: E402


def resolve_glob(pattern):
    """Resolve a glob pattern to a single path."""
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        print(f"ERROR: No match for {pattern}")
        sys.exit(1)
    return Path(matches[-1])


def load_model_and_adapter(adapter_path):
    """Load base model with the saved adapter."""
    from experiments.utils.test_harness import get_benchmark_models, load_model_and_config

    # Infer model from adapter config
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            cfg = json.load(f)
        base_model_id = cfg.get("base_model_name_or_path", "")
        print(f"  Base model: {base_model_id}")

    # Load base model (Mistral)
    parser = argparse.ArgumentParser()
    from experiments.utils.test_harness import add_model_args

    add_model_args(parser)
    dummy_args = parser.parse_args(["--model", "mistral"])
    models = list(get_benchmark_models(dummy_args))
    bench_name, bench_config = models[0]

    model, tokenizer = load_model_and_config(bench_config)

    # Load saved adapter
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name="episodic")
    model.set_adapter("episodic")
    model.gradient_checkpointing_disable()

    print(f"  Adapter loaded: {adapter_path}")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.0):
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Interactive adapter probe")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to saved adapter directory (supports globs)",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        help="Path to keyed_pairs.json (supports globs) — shows available keys",
    )
    args = parser.parse_args()

    adapter_path = resolve_glob(args.adapter)

    keyed_pairs = None
    if args.keys:
        keys_path = resolve_glob(args.keys)
        with open(keys_path) as f:
            keyed_pairs = json.load(f)

    print("\nLoading model and adapter...")
    model, tokenizer = load_model_and_adapter(adapter_path)

    if keyed_pairs:
        print(f"\n  {len(keyed_pairs)} keys available (graph1 .. graph{len(keyed_pairs)})")
        print(f"  Sample: {keyed_pairs[0]['key']} -> {keyed_pairs[0]['question']}")

    print("\nReady. Type prompts to query the adapter.")
    print("  Special commands:")
    print("    /key <N>     — recall key graphN")
    print("    /list        — show all keys")
    print("    /list <term> — search keys")
    print("    /quit        — exit")
    print()

    while True:
        try:
            prompt = input("probe> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        if prompt == "/quit":
            break

        if prompt == "/list":
            if keyed_pairs:
                for kp in keyed_pairs:
                    print(f"  {kp['key']}: {kp['question']} -> {kp['answer'][:60]}")
            else:
                print("  No keyed_pairs loaded. Use --keys to load.")
            continue

        if prompt.startswith("/list "):
            term = prompt[6:].lower()
            if keyed_pairs:
                for kp in keyed_pairs:
                    if term in kp["question"].lower() or term in kp["answer"].lower():
                        print(f"  {kp['key']}: {kp['question']} -> {kp['answer'][:60]}")
            continue

        if prompt.startswith("/key "):
            key_num = prompt[5:].strip()
            prompt = f"Recall the QA pair stored under key 'graph{key_num}'."

        print(f"  [{prompt[:80]}{'...' if len(prompt) > 80 else ''}]")
        response = generate(model, tokenizer, prompt)
        print(f"\n{response}\n")

    unload_model(model, tokenizer)


if __name__ == "__main__":
    main()
