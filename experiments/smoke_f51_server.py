"""Phase 5.1 GPU smoke test — end-to-end server pipeline without HTTP.

Validates the full cycle:
1. Load model (no adapter yet)
2. Chat with base model (should answer generically)
3. Buffer a conversation session with personal facts
4. Run consolidation (extract → graph → QA gen → train adapter)
5. Chat again (adapter active — should recall trained facts)
6. Verify recall via key probing

Usage:
    python experiments/smoke_f51_server.py --model gemma
    python experiments/smoke_f51_server.py --model mistral
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

from paramem.models.loader import load_base_model, unload_model  # noqa: E402
from paramem.server.config import ServerConfig, ServerTrainingConfig  # noqa: E402
from paramem.server.consolidation import run_consolidation  # noqa: E402
from paramem.server.inference import handle_chat  # noqa: E402
from paramem.server.session_buffer import SessionBuffer  # noqa: E402
from paramem.training.indexed_memory import probe_key  # noqa: E402

# Smoke test data — 5 clear personal facts
SMOKE_CONVERSATIONS = [
    {
        "session_id": "smoke_session_1",
        "turns": [
            ("user", "My name is Marcus and I work at CERN as a particle physicist."),
            ("assistant", "Nice to meet you, Marcus! That's fascinating work at CERN."),
            ("user", "I live in Geneva with my wife Elena and our cat Schrödinger."),
            ("assistant", "Geneva is wonderful. And what a perfect name for a physicist's cat!"),
            ("user", "On weekends I play chess at the local club. I'm rated about 1800 ELO."),
            ("assistant", "1800 ELO is quite strong! Do you compete in tournaments?"),
        ],
    },
]

# Questions to test after consolidation
RECALL_QUESTIONS = [
    "Where does Marcus work?",
    "What is the name of Marcus's cat?",
    "What does Marcus do on weekends?",
]


def run_smoke_test(model_name: str):
    """Run the full Phase 5.1 smoke test."""
    smoke_dir = project_root / "outputs" / "smoke_f51" / model_name
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    config = ServerConfig(
        model_name=model_name,
        adapter_dir=smoke_dir / "adapters",
        registry_path=smoke_dir / "registry.json",
        graph_path=smoke_dir / "graph.json",
        session_dir=smoke_dir / "sessions",
        training=ServerTrainingConfig(epochs=30, rank=8, alpha=16),
    )
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.model_config
    logger.info("=== Phase 5.1 Smoke Test: %s ===", model_name)

    # Step 1: Load model
    logger.info("Step 1: Loading model %s", model_config.model_id)
    model, tokenizer = load_base_model(model_config)
    logger.info("Model loaded")

    # Step 2: Chat without adapter (baseline)
    logger.info("Step 2: Chat without adapter (baseline)")
    result = handle_chat(
        text="Where does Marcus work?",
        conversation_id="test",
        history=None,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    logger.info("  Base model response: %s", result.text[:200])

    # Step 3: Buffer conversation sessions
    logger.info("Step 3: Buffering conversation sessions")
    buffer = SessionBuffer(config.session_dir)
    for session in SMOKE_CONVERSATIONS:
        for role, text in session["turns"]:
            buffer.append(session["session_id"], role, text)
    logger.info(
        "  Buffered %d sessions, %d pending",
        len(SMOKE_CONVERSATIONS),
        buffer.pending_count,
    )

    # Step 4: Run consolidation
    logger.info("Step 4: Running consolidation")
    start = time.time()
    consolidation_result = run_consolidation(
        model=model,
        tokenizer=tokenizer,
        config=config,
        session_buffer=buffer,
    )
    elapsed = time.time() - start
    logger.info("  Consolidation result: %s", consolidation_result)
    logger.info("  Elapsed: %.1fs", elapsed)

    if consolidation_result.get("status") != "complete":
        logger.error("Consolidation failed: %s", consolidation_result)
        unload_model(model, tokenizer)
        return False

    # Step 5: Chat with adapter active
    logger.info("Step 5: Chat with adapter (should recall facts)")
    results = {}
    for question in RECALL_QUESTIONS:
        result = handle_chat(
            text=question,
            conversation_id="test_recall",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        results[question] = result.text
        logger.info("  Q: %s", question)
        logger.info("  A: %s", result.text[:200])

    # Step 6: Verify via key probing
    logger.info("Step 6: Key probing verification")
    if config.registry_path.exists():
        registry_data = json.loads(config.registry_path.read_text())
        registry = {}
        for key, meta in registry_data.items():
            if isinstance(meta, dict):
                registry[key] = meta.get("simhash", 0)

        total_keys = len(registry)
        recalled = 0
        for key in sorted(registry.keys()):
            probe = probe_key(model, tokenizer, key, registry=registry)
            if probe and "failure_reason" not in probe:
                recalled += 1
                logger.info("  %s: OK — Q: %s", key, probe.get("question", "")[:60])
            else:
                reason = probe.get("failure_reason", "unknown") if probe else "empty"
                logger.info("  %s: FAIL — %s", key, reason)

        logger.info("  Key recall: %d/%d", recalled, total_keys)
    else:
        logger.warning("  No registry found — skipping key probe")
        total_keys = 0
        recalled = 0

    # Save results
    output = {
        "model": model_name,
        "consolidation": consolidation_result,
        "consolidation_time": round(elapsed, 1),
        "chat_responses": results,
        "key_recall": {"recalled": recalled, "total": total_keys},
    }
    results_path = smoke_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", results_path)

    unload_model(model, tokenizer)

    # Summary
    print()
    print("=" * 60)
    print(f"  PHASE 5.1 SMOKE TEST: {model_name}")
    print("=" * 60)
    print(f"  Consolidation: {consolidation_result.get('status')}")
    print(f"  Keys trained:  {consolidation_result.get('total_keys', 0)}")
    print(f"  Train loss:    {consolidation_result.get('train_loss', '?')}")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"  Key recall:    {recalled}/{total_keys}")
    print()
    for q, a in results.items():
        print(f"  Q: {q}")
        print(f"  A: {a[:100]}")
        print()
    print("=" * 60)

    return consolidation_result.get("status") == "complete" and recalled > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5.1 GPU Smoke Test")
    parser.add_argument("--model", default="gemma", choices=["gemma", "mistral"])
    args = parser.parse_args()

    success = run_smoke_test(args.model)
    sys.exit(0 if success else 1)
