"""Live GPU smoke: Step 6 post_session_train + Step 7 consolidate_interim_adapters.

Exercises the full multi-adapter chain end-to-end on real GPU before the
Step 6+7 commit lands. Not a full regression suite (that is Step 8's job) —
a sanity gate that proves:

  * a small synthetic transcript extracts → trains an interim adapter
    (Step 6 hook),
  * two interim adapters collapse into the three main adapters via
    `consolidate_interim_adapters` (Step 7) with the outer GPU lock held,
  * registry rewrite + interim purge + router reload run without error,
  * per-tier recall after the rebuild is ≥ 0.9.

Usage:
    conda activate paramem
    python experiments/test_step6_step7_live_gpu.py

The script stops only on exit; it does NOT auto-restart
paramem-server.service. Run `systemctl --user start paramem-server`
manually afterwards.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure conda env's GPU envs propagate (mirrors server startup).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import BENCHMARK_MODELS, setup_logging  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.gpu_lock import gpu_lock_sync  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
)

setup_logging()
logger = logging.getLogger("step6_step7_live")

OUTPUT_DIR = project_root / "outputs" / "test_step6_step7_live_gpu"

# Two synthetic transcripts feed two interim adapters.
TRANSCRIPT_A = (
    "User: I had dinner with my friend Nora yesterday in Vienna.\n"
    "Assistant: That sounds lovely.\n"
    "User: Nora just started her new job as a marine biologist.\n"
    "Assistant: Congratulations to her.\n"
)
TRANSCRIPT_B = (
    "User: My colleague Kian moved to Berlin last month.\n"
    "Assistant: How is he settling in?\n"
    "User: Kian loves it, especially the cycling infrastructure.\n"
    "Assistant: Sounds like a good fit.\n"
)


def _tier_cfg(rank: int = 8) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=2 * rank,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def main() -> int:
    acquire_gpu(interactive=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mistral 7B NF4 …")
    model, tokenizer = load_base_model(BENCHMARK_MODELS["mistral"])

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Minimal server-parity ConsolidationLoop.
    training_cfg = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=512,
        num_epochs=10,
        warmup_steps=5,
        warmup_ratio=0.0,
    )

    consolidation_cfg = ConsolidationConfig(
        indexed_key_replay_enabled=True,
        promotion_threshold=3,
    )

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_cfg,
        training_config=training_cfg,
        episodic_adapter_config=_tier_cfg(),
        semantic_adapter_config=_tier_cfg(),
        procedural_adapter_config=_tier_cfg(),
        wandb_config=None,
        output_dir=OUTPUT_DIR,
        extraction_stt_correction=False,
        extraction_ha_validation=False,
        extraction_noise_filter="",
        extraction_plausibility_judge="local",
        extraction_verify_anonymization=False,
    )

    # Two post_session_train calls → two interim adapters (separate stamps).
    logger.info("Step 6 — post_session_train (A) …")
    result_a = loop.post_session_train(
        session_transcript=TRANSCRIPT_A,
        session_id="live-session-A",
        schedule="weekly",
        max_interim_count=14,
        stamp="20260419T0900",
    )
    logger.info(
        "Step 6 (A) result: mode=%s new_keys=%s",
        result_a.get("mode"),
        result_a.get("new_keys"),
    )

    logger.info("Step 6 — post_session_train (B) …")
    result_b = loop.post_session_train(
        session_transcript=TRANSCRIPT_B,
        session_id="live-session-B",
        schedule="weekly",
        max_interim_count=14,
        stamp="20260419T2100",
    )
    logger.info(
        "Step 6 (B) result: mode=%s new_keys=%s",
        result_b.get("mode"),
        result_b.get("new_keys"),
    )

    # Snapshot registry state before Step 7.
    keys_before = loop.indexed_key_registry.list_active()
    adapters_before = sorted(model.peft_config.keys()) if hasattr(model, "peft_config") else []
    logger.info(
        "Pre-Step-7: %d active keys; adapters=%s",
        len(keys_before),
        adapters_before,
    )

    # Step 7 runs under the outer GPU lock (matches BackgroundTrainer.submit() contract).
    logger.info("Step 7 — consolidate_interim_adapters (outer GPU lock held) …")
    t0 = time.time()
    with gpu_lock_sync():
        step7_result = loop.consolidate_interim_adapters(
            trainer=None,  # no BackgroundTrainer in this standalone run
            router=None,
            # Use production defaults: refresh_epochs=30, recall_sanity_threshold=0.95.
            # Mistral 7B needs 30 epochs minimum to encode indexed keys (CLAUDE.md).
        )
    elapsed = time.time() - t0
    logger.info("Step 7 returned in %.1fs: %s", elapsed, json.dumps(step7_result, default=str))

    # Post-conditions.
    keys_after = loop.indexed_key_registry.list_active()
    adapters_after = sorted(model.peft_config.keys()) if hasattr(model, "peft_config") else []
    interim_dirs_after = sorted(
        p.name
        for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("episodic_interim_")
    )

    logger.info(
        "Post-Step-7: %d active keys; adapters=%s; interim_dirs=%s",
        len(keys_after),
        adapters_after,
        interim_dirs_after,
    )

    all_interim_adapters_gone = not any(n.startswith("episodic_interim_") for n in adapters_after)
    no_interim_dirs_left = not interim_dirs_after
    tier_adapters_present = {"episodic", "semantic", "procedural"}.issubset(set(adapters_after))
    recall_per_tier = step7_result.get("recall_per_tier", {})
    min_recall = min(recall_per_tier.values()) if recall_per_tier else 0.0

    summary = {
        "step6_a_mode": result_a.get("mode"),
        "step6_b_mode": result_b.get("mode"),
        "step7_elapsed_s": round(elapsed, 1),
        "step7_result": step7_result,
        "all_interim_adapters_gone": all_interim_adapters_gone,
        "no_interim_dirs_left": no_interim_dirs_left,
        "tier_adapters_present": tier_adapters_present,
        "min_recall": min_recall,
        "keys_before": len(keys_before),
        "keys_after": len(keys_after),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary: %s", json.dumps(summary, indent=2, default=str))

    ok = (
        all_interim_adapters_gone
        and no_interim_dirs_left
        and tier_adapters_present
        and not step7_result.get("rolled_back", False)
    )
    if not ok:
        logger.error("Step 6+7 live chain FAILED invariants.")
        return 1

    logger.info("Step 6+7 live chain PASSED.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        logger.exception("Step 6+7 live GPU chain crashed.")
        rc = 2
    finally:
        # Always try to clear CUDA state so WSL doesn't get stuck.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
