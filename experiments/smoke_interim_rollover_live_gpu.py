"""Live GPU smoke for the interim-rollover mini-enrichment hook.

Exercises the new `post_session_train` hook end-to-end on real hardware:

  * Real Mistral 7B NF4 on GPU.
  * Real Anthropic SOTA for noise-filter extraction AND graph enrichment.
  * Two synthetic sessions with distinct stamps → two interim adapters.
  * Test 8's cumulative graph preloaded as the "background knowledge"
    so the enrichment floor (≥10 nodes) is already crossed on session A.

Assertions:
  1. Session A (new stamp) triggers the rollover hook → non-skipped
     enrichment → counter reset to 0.
  2. Session B (different stamp, new interim) re-fires the hook after
     the counter re-crosses the floor.
  3. Both interim adapters exist in peft_config after training.
  4. At least one new edge tagged `source="graph_enrichment"` lands on
     the cumulative graph during the smoke.

Pre-conditions:
  * `paramem-server.service` STOPPED — we need the GPU.
  * `export $(grep -v '^#' .env | xargs)` → ANTHROPIC_API_KEY available.

Usage:
    conda activate paramem
    python experiments/smoke_interim_rollover_live_gpu.py

The script does NOT restart the server — run
`systemctl --user start paramem-server` afterwards.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx  # noqa: E402
import torch  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import BENCHMARK_MODELS, setup_logging  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402

setup_logging()
logger = logging.getLogger("smoke_interim_rollover_live_gpu")

OUTPUT_DIR = project_root / "outputs" / "smoke_interim_rollover_live_gpu"
TEST8_GRAPH = project_root / (
    "outputs/test8_large_scale/mistral/20260323_161747/cycle_056/cumulative_graph.json"
)

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


def _count_enrichment_edges(graph: nx.MultiDiGraph) -> int:
    return sum(1 for _, _, d in graph.edges(data=True) if d.get("source") == "graph_enrichment")


def main() -> int:
    acquire_gpu(interactive=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set — aborting.")
        return 3
    if not TEST8_GRAPH.exists():
        logger.error("Test 8 graph missing at %s — aborting.", TEST8_GRAPH)
        return 4

    logger.info("Loading Mistral 7B NF4 …")
    model, tokenizer = load_base_model(BENCHMARK_MODELS["mistral"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    from paramem.server.config import load_server_config
    from paramem.server.consolidation import create_consolidation_loop

    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.model_name = "mistral"

    for tier in (cfg.adapters.episodic, cfg.adapters.semantic, cfg.adapters.procedural):
        tier.enabled = True
        tier.rank = 8
        tier.alpha = 16
        tier.learning_rate = 1e-4
        tier.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    cfg.consolidation.max_epochs = 10
    cfg.consolidation.indexed_key_replay = True
    cfg.consolidation.promotion_threshold = 3

    cfg.consolidation.extraction_stt_correction = False
    cfg.consolidation.extraction_ha_validation = False
    # Anthropic SOTA for noise filter AND graph enrichment (same creds).
    cfg.consolidation.extraction_noise_filter = "anthropic"
    cfg.consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
    cfg.consolidation.extraction_plausibility_judge = "off"
    cfg.consolidation.extraction_verify_anonymization = False

    # Interim-rollover mini-enrichment ON, small budgets to keep the smoke cheap.
    cfg.consolidation.graph_enrichment_enabled = True
    cfg.consolidation.graph_enrichment_neighborhood_hops = 1
    cfg.consolidation.graph_enrichment_max_entities_per_pass = 400
    cfg.consolidation.graph_enrichment_interim_enabled = True
    cfg.consolidation.graph_enrichment_min_triples_floor = 1

    loop = create_consolidation_loop(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        state_provider=None,
        output_dir=OUTPUT_DIR,
        save_cycle_snapshots=False,
        persist_graph=False,
        seed_state_from_disk=False,
    )

    # Preload Test 8's cumulative graph so the 10-node floor is crossed
    # from the first session. Real ConsolidationLoop.merger is a live
    # GraphMerger; its underlying nx graph is `.graph`.
    logger.info("Preloading Test 8 cumulative graph into merger …")
    with open(TEST8_GRAPH) as f:
        preloaded = nx.node_link_graph(json.load(f))
    loop.merger.graph = preloaded
    n_nodes_initial = preloaded.number_of_nodes()
    n_edges_initial = preloaded.number_of_edges()
    enrichment_edges_initial = _count_enrichment_edges(preloaded)
    logger.info(
        "  preloaded: nodes=%d edges=%d (already tagged graph_enrichment=%d)",
        n_nodes_initial,
        n_edges_initial,
        enrichment_edges_initial,
    )

    # --- Session A: first stamp → normal fresh-interim branch → hook fires ---
    counter_before_a = loop._triples_since_last_enrichment
    logger.info("Session A: post_session_train (stamp=20260419T0900)")
    t_a = time.time()
    result_a = loop.post_session_train(
        session_transcript=TRANSCRIPT_A,
        session_id="smoke-A",
        schedule="12h",
        max_interim_count=7,
        stamp="20260419T0900",
    )
    elapsed_a = time.time() - t_a
    counter_after_a = loop._triples_since_last_enrichment
    enrichment_edges_after_a = _count_enrichment_edges(loop.merger.graph)
    logger.info(
        "  A: mode=%s counter=%d→%d elapsed=%.1fs enrichment_edges=%d",
        result_a.get("mode"),
        counter_before_a,
        counter_after_a,
        elapsed_a,
        enrichment_edges_after_a,
    )

    # --- Session B: new stamp → new interim → hook fires again (if floor re-crossed) ---
    counter_before_b = loop._triples_since_last_enrichment
    logger.info("Session B: post_session_train (stamp=20260419T2100)")
    t_b = time.time()
    result_b = loop.post_session_train(
        session_transcript=TRANSCRIPT_B,
        session_id="smoke-B",
        schedule="12h",
        max_interim_count=7,
        stamp="20260419T2100",
    )
    elapsed_b = time.time() - t_b
    counter_after_b = loop._triples_since_last_enrichment
    enrichment_edges_after_b = _count_enrichment_edges(loop.merger.graph)
    logger.info(
        "  B: mode=%s counter=%d→%d elapsed=%.1fs enrichment_edges=%d",
        result_b.get("mode"),
        counter_before_b,
        counter_after_b,
        elapsed_b,
        enrichment_edges_after_b,
    )

    # --- Post-conditions ---
    adapter_names = sorted(model.peft_config.keys()) if hasattr(model, "peft_config") else []
    interim_names = sorted(n for n in adapter_names if n.startswith("episodic_interim_"))

    failures: list[str] = []
    if result_a.get("mode") != "trained":
        failures.append(f"Session A mode={result_a.get('mode')} (expected 'trained')")
    if result_b.get("mode") != "trained":
        failures.append(f"Session B mode={result_b.get('mode')} (expected 'trained')")
    if counter_after_a != 0:
        failures.append(
            f"Session A hook did NOT reset counter (={counter_after_a}) — enrichment skipped?"
        )
    if counter_after_b != 0:
        failures.append(
            f"Session B hook did NOT reset counter (={counter_after_b}) — enrichment skipped?"
        )
    if len(interim_names) < 2:
        failures.append(f"expected 2 interim adapters, found {len(interim_names)}: {interim_names}")
    new_enrichment = enrichment_edges_after_b - enrichment_edges_initial
    if new_enrichment < 1:
        failures.append(
            f"no new graph_enrichment edges landed (delta={new_enrichment}) "
            "— SOTA may have returned empty for both chunks"
        )

    summary = {
        "model": "mistral",
        "preloaded_graph": str(TEST8_GRAPH),
        "nodes_initial": n_nodes_initial,
        "edges_initial": n_edges_initial,
        "enrichment_edges_initial": enrichment_edges_initial,
        "enrichment_edges_after_a": enrichment_edges_after_a,
        "enrichment_edges_after_b": enrichment_edges_after_b,
        "session_a": {
            "mode": result_a.get("mode"),
            "new_keys": result_a.get("new_keys"),
            "adapter": result_a.get("adapter_name"),
            "counter_before": counter_before_a,
            "counter_after": counter_after_a,
            "elapsed_s": round(elapsed_a, 1),
        },
        "session_b": {
            "mode": result_b.get("mode"),
            "new_keys": result_b.get("new_keys"),
            "adapter": result_b.get("adapter_name"),
            "counter_before": counter_before_b,
            "counter_after": counter_after_b,
            "elapsed_s": round(elapsed_b, 1),
        },
        "adapter_names_after": adapter_names,
        "interim_names_after": interim_names,
        "failures": failures,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary:\n%s", json.dumps(summary, indent=2, default=str))

    if failures:
        logger.error("Interim-rollover GPU smoke FAILED:")
        for f in failures:
            logger.error("  - %s", f)
        return 1
    logger.info("Interim-rollover GPU smoke PASSED.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        logger.exception("Interim-rollover GPU smoke crashed.")
        rc = 2
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
