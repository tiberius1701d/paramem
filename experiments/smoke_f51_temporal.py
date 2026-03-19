"""Phase 5.1 temporal query smoke test.

Loads the adapter from the Phase 5 smoke test and asks temporal questions
to verify the registry-based key lookup → probe → context → answer pipeline.

Requires: outputs/smoke_f51/gemma/ from a prior smoke_f51_server.py run.

Usage:
    python experiments/smoke_f51_temporal.py
"""

import logging
import os
import sys
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

from paramem.models.loader import load_adapter, load_base_model, unload_model  # noqa: E402
from paramem.server.config import ServerConfig  # noqa: E402
from paramem.server.inference import handle_chat  # noqa: E402
from paramem.server.temporal import detect_temporal_query  # noqa: E402

SMOKE_DIR = project_root / "outputs" / "smoke_f51" / "gemma"

TEMPORAL_QUESTIONS = [
    "What did we discuss today?",
    "What do you remember from earlier today?",
    "Remind me what we talked about this morning.",
    "What did we discuss yesterday?",  # no keys match — should fall back
]

DIRECT_QUESTIONS = [
    "Where does Marcus work?",
    "What is Marcus's cat called?",
]


def main():
    if not (SMOKE_DIR / "adapters" / "episodic").exists():
        print("ERROR: No adapter found. Run smoke_f51_server.py first.")
        sys.exit(1)

    config = ServerConfig(
        model_name="gemma",
        adapter_dir=SMOKE_DIR / "adapters",
        registry_path=SMOKE_DIR / "registry.json",
        graph_path=SMOKE_DIR / "graph.json",
        session_dir=SMOKE_DIR / "sessions",
    )

    # Verify temporal detection works before loading model
    print("=== Temporal detection (no GPU) ===")
    for q in TEMPORAL_QUESTIONS + DIRECT_QUESTIONS:
        result = detect_temporal_query(q)
        label = f"TEMPORAL {result}" if result else "STANDARD"
        print(f"  [{label}] {q}")

    # Load model + adapter
    print("\n=== Loading model + adapter ===")
    model, tokenizer = load_base_model(config.model_config)
    model = load_adapter(model, str(config.adapter_dir), "episodic")
    print("  Model and adapter loaded")

    # Test temporal queries
    print("\n=== Temporal queries (registry lookup → probe → reason) ===")
    for q in TEMPORAL_QUESTIONS:
        result = handle_chat(
            text=q,
            conversation_id="temporal_test",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        print(f"  Q: {q}")
        print(f"  A: {result.text[:200]}")
        if result.probed_keys:
            print(f"  Keys probed: {result.probed_keys}")
        print()

    # Test direct queries (should use standard path)
    print("=== Direct queries (adapter active, no retrieval) ===")
    for q in DIRECT_QUESTIONS:
        result = handle_chat(
            text=q,
            conversation_id="direct_test",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        print(f"  Q: {q}")
        print(f"  A: {result.text[:200]}")
        print()

    unload_model(model, tokenizer)
    print("=== Temporal smoke test complete ===")


if __name__ == "__main__":
    main()
