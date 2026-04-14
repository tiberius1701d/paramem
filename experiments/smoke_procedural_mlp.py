"""Smoke test: procedural adapter trains with attention + MLP target modules.

Validates that:
1. `ServerConfig` honours per-adapter `target_modules` — procedural picks up
   attention + MLP (`q/k/v/o/gate/up/down_proj`).
2. The PEFT model actually creates LoRA modules on MLP layers.
3. A tiny training pass converges (loss drops) without CUDA OOM on 8 GB VRAM.

Run on a GPU host. Exits non-zero on any regression.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.config import ServerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
EXPECTED_ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def main() -> int:
    # 1. Verify config plumbing.
    cfg = ServerConfig()
    proc_targets = set(cfg.adapters.procedural.target_modules)
    logger.info("procedural target_modules: %s", sorted(proc_targets))
    assert EXPECTED_ATTN_MODULES.issubset(proc_targets), (
        f"procedural missing attention modules: {EXPECTED_ATTN_MODULES - proc_targets}"
    )
    assert EXPECTED_MLP_MODULES.issubset(proc_targets), (
        f"procedural missing MLP modules: {EXPECTED_MLP_MODULES - proc_targets}"
    )

    epi_targets = set(cfg.adapters.episodic.target_modules)
    assert epi_targets == EXPECTED_ATTN_MODULES, (
        f"episodic should stay attention-only, got {sorted(epi_targets)}"
    )

    adapter_cfg = cfg._make_adapter_config(cfg.adapters.procedural)
    assert set(adapter_cfg.target_modules) == proc_targets

    # 2. Load base model (Mistral 7B NF4 per server default).
    logger.info("Loading base model: %s", cfg.model_config.model_id)
    vram_before = torch.cuda.memory_allocated() / 1e9
    model, tokenizer = load_base_model(cfg.model_config)
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    logger.info(
        "VRAM after base model load: %.2f GB (Δ %.2f)",
        vram_after_load,
        vram_after_load - vram_before,
    )

    # 3. Attach procedural LoRA adapter with MLP targeting.
    lora = LoraConfig(
        r=adapter_cfg.rank,
        lora_alpha=adapter_cfg.alpha,
        target_modules=adapter_cfg.target_modules,
        lora_dropout=adapter_cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    peft_model.print_trainable_parameters()
    vram_after_adapter = torch.cuda.memory_allocated() / 1e9
    logger.info(
        "VRAM after adapter attach: %.2f GB (Δ %.2f)",
        vram_after_adapter,
        vram_after_adapter - vram_after_load,
    )

    # 4. Verify LoRA modules exist on both attention AND MLP layers.
    attn_lora_hits = 0
    mlp_lora_hits = 0
    for name, _ in peft_model.named_modules():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if any(m in name for m in EXPECTED_MLP_MODULES):
            mlp_lora_hits += 1
        if any(m in name for m in EXPECTED_ATTN_MODULES):
            attn_lora_hits += 1
    logger.info("LoRA modules attached — attention: %d, MLP: %d", attn_lora_hits, mlp_lora_hits)
    assert attn_lora_hits > 0, "no LoRA on attention layers"
    assert mlp_lora_hits > 0, "no LoRA on MLP layers (THIS IS THE WHOLE POINT)"

    # 5. Memorization check: train on the SAME sentence 10 steps and confirm
    # loss drops materially. Cross-sample loss comparison is noise; same-sample
    # repeated loss is a clean convergence signal for a LoRA adapter.
    text = "The user prefers concise answers in the morning."
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(
        "cuda"
    )
    labels = enc["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=adapter_cfg.learning_rate
    )
    peft_model.train()
    losses = []
    vram_peak = 0.0
    for step in range(10):
        t0 = time.time()
        optimizer.zero_grad()
        out = peft_model(**enc, labels=labels)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        vram_peak = max(vram_peak, torch.cuda.max_memory_allocated() / 1e9)
        losses.append(out.loss.item())
        logger.info(
            "step %d loss=%.4f (%.2fs, peak VRAM %.2f GB)",
            step,
            losses[-1],
            time.time() - t0,
            vram_peak,
        )

    # 6. Assertions.
    assert losses[-1] < losses[0] * 0.6, (
        f"loss did not drop enough (>40% reduction expected): {losses}"
    )
    budget_gb = 7.5  # leave headroom on 8 GB
    assert vram_peak < budget_gb, f"peak VRAM {vram_peak:.2f} GB exceeds {budget_gb} GB budget"

    logger.info(
        "SMOKE PASS — procedural MLP adapter trains cleanly. losses=%s peak=%.2f GB",
        losses,
        vram_peak,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
