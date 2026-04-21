"""Live-GPU probe for the no-hallucination fallback package.

Exercises the full production _probe_and_reason path against a real
trained adapter to verify three contracts end-to-end:

  1. In-domain recall — the model surfaces facts present in the
     layered_context assembled from recalled keys.
  2. Anti-confabulation (Slice 2) — off-domain personal queries
     (location, job, birthday, spouse) return polite abstentions
     instead of invented facts.
  3. Escalation sentinel — real-time queries (time, weather) and
     general-knowledge off-domain queries emit `[ESCALATE]` somewhere
     in the response so paramem.server.escalation.detect_escalation
     routes them to HA/SOTA.

This is the live-GPU counterpart of tests/test_voice_prompt_contract.py
(which is string-level only). It mirrors production exactly:
  probe_keys_grouped_by_adapter → layered_context
  → _personalize_prompt(voice.load_prompt(), speaker)
  → model.disable_adapter() generate
  → detect_escalation

Subject: Test 8 cycle_001 Mistral adapter, speaker "Xiaoyu"
(8 predicates covering interests/entertainment — small enough to
make "off-domain" queries unambiguous).

Outputs to outputs/voice_prompt_contract/<timestamp>/results.json.
No existing training data is touched; no schema changes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    setup_logging,
)
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.models.loader import load_adapter, load_base_model  # noqa: E402
from paramem.server.config import load_server_config  # noqa: E402
from paramem.server.escalation import detect_escalation  # noqa: E402
from paramem.server.inference import _personalize_prompt  # noqa: E402
from paramem.training.indexed_memory import probe_keys_grouped_by_adapter  # noqa: E402

CYCLE_DIR = (
    PROJECT_ROOT / "outputs" / "test8_large_scale" / "mistral" / "20260323_161747" / "cycle_001"
)
SUBJECT = "Xiaoyu"
ADAPTER_NAME = "episodic"

# First-person — speaker=Xiaoyu via _personalize_prompt (production HA /chat path)
IN_DOMAIN_QUERIES = [
    "What movie did I watch recently?",
    "What am I interested in?",
]
# Personal-but-off-domain — sanitizer would block SOTA escalation
OFF_DOMAIN_PERSONAL_QUERIES = [
    "Where do I live?",
    "What is my job?",
    "When was I born?",
    "Who is my spouse?",
]
# Sanitizer-safe general knowledge — [ESCALATE] should fire so SOTA answers
SANITIZER_SAFE_GENERAL_QUERIES = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "How many continents are there?",
]
# Real-time queries — prompt explicitly enumerates these as escalation triggers
REALTIME_QUERIES = [
    "What time is it right now?",
    "What is the weather like today?",
]


def main() -> int:
    setup_logging()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "voice_prompt_contract" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load keyed_pairs + registry
    with open(CYCLE_DIR / "keyed_pairs.json") as f:
        keyed_pairs = json.load(f)
    with open(CYCLE_DIR / "simhash_registry.json") as f:
        registry = json.load(f)

    subject_keys = [kp["key"] for kp in keyed_pairs if kp["source_subject"] == SUBJECT]
    print(f"Subject '{SUBJECT}': {len(subject_keys)} trained keys: {subject_keys}")

    # Load server config (for voice prompt — the Slice-2 version)
    server_config = load_server_config()
    voice_prompt_raw = server_config.voice.load_prompt()
    system_prompt = _personalize_prompt(voice_prompt_raw, SUBJECT, None, server_config)
    print(f"\nSystem prompt (len={len(system_prompt)}):\n---\n{system_prompt}\n---")

    # Load base + adapter
    model_config = BENCHMARK_MODELS["mistral"]
    model, tokenizer = load_base_model(model_config)
    adapter_dir = CYCLE_DIR / "adapter"
    model = load_adapter(model, adapter_dir, ADAPTER_NAME)

    # Probe all keys for this subject — reproduces _probe_and_reason
    probe_results = probe_keys_grouped_by_adapter(
        model, tokenizer, {ADAPTER_NAME: subject_keys}, registry=registry
    )
    layer_facts = []
    for key in subject_keys:
        r = probe_results.get(key)
        if r and "failure_reason" not in r:
            layer_facts.append(f"- {r.get('answer', '')}")
    layered_context = "[Recent knowledge]\n" + "\n".join(layer_facts)
    print(f"\nRecalled {len(layer_facts)} facts for layered_context.")

    # Run the probe
    from peft import PeftModel

    results = []

    def run_query(kind: str, query: str):
        augmented = f"What you know about the speaker:\n\n{layered_context}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                response = generate_answer(
                    model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
                )
        else:
            response = generate_answer(
                model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
            )
        escalated, forwarded = detect_escalation(response)
        row = {
            "kind": kind,
            "query": query,
            "response": response,
            "escalated": escalated,
            "forwarded_query": forwarded,
        }
        results.append(row)
        marker = "ESCALATE→SOTA" if escalated else "ANSWERED"
        print(f"\n[{kind}] Q: {query}\n  → [{marker}] {response!r}")

    for q in IN_DOMAIN_QUERIES:
        run_query("in_domain", q)
    for q in OFF_DOMAIN_PERSONAL_QUERIES:
        run_query("off_domain_personal", q)
    for q in SANITIZER_SAFE_GENERAL_QUERIES:
        run_query("sanitizer_safe_general", q)
    for q in REALTIME_QUERIES:
        run_query("realtime", q)

    def counts(kind: str) -> tuple[int, int]:
        e = sum(1 for r in results if r["kind"] == kind and r["escalated"])
        t = sum(1 for r in results if r["kind"] == kind)
        return e, t

    summary = {}
    for kind in (
        "in_domain",
        "off_domain_personal",
        "sanitizer_safe_general",
        "realtime",
    ):
        e, t = counts(kind)
        summary[kind] = {
            "escalated": f"{e}/{t}",
            "rate": (e / t) if t else 0.0,
        }

    report = {
        "cycle_dir": str(CYCLE_DIR),
        "subject": SUBJECT,
        "subject_keys": subject_keys,
        "recalled_facts": layer_facts,
        "system_prompt": system_prompt,
        "summary": summary,
        "results": results,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSummary: {report['summary']}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    with acquire_gpu():
        sys.exit(main())
