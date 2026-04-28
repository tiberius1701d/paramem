"""Quick smoke test: run 3 consolidation cycles to validate pipeline."""

import json
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["WANDB_MODE"] = "disabled"

from paramem.models.loader import create_adapter, load_base_model  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import TrainingConfig, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    episodic_config = config.adapters["episodic"]
    semantic_config = config.adapters["semantic"]
    model = create_adapter(model, episodic_config, "episodic")
    model = create_adapter(model, semantic_config, "semantic")

    # Load first 3 sessions
    sessions_path = project_root / "data" / "synthetic" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)[:3]
    logger.info("Loaded %d sessions for smoke test", len(sessions))

    # Minimal training config: 3 epochs, fast
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=512,
        num_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=config.consolidation,
        training_config=training_config,
        episodic_adapter_config=episodic_config,
        semantic_adapter_config=semantic_config,
        output_dir=project_root / "outputs" / "phase3_smoke",
        extraction_temperature=config.graph.extraction_temperature,
    )

    for session in sessions:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        logger.info(
            "Cycle %d: %d entities, %d relations, %d promoted, %d decayed (%.1fs)",
            result.cycle_index,
            result.entities_extracted,
            result.relations_extracted,
            result.nodes_promoted,
            result.nodes_decayed,
            result.wall_clock_seconds,
        )

    logger.info("Smoke test PASSED - pipeline works end-to-end")


if __name__ == "__main__":
    main()
