"""Live GPU probe: extraction + graph-merge pipeline validation.

Validates the uncommitted working-tree code path:
  transcript → ExtractionPipeline.run() → GraphMerger.merge()

Two fictional sessions with a SAME-PREDICATE residence conflict (Munich →
Berlin) and additive pet facts are extracted and merged.  The same-predicate
conflict is designed to trigger the merger's cardinality resolution
(``check_predicate_coexistence``), which loads the EXTERNALISED coexistence
prompt and makes a live model REPLACE/COEXIST decision — closing the coverage
gap where the externalised prompt is loaded but never exercised.

Assertions are written to a timestamped debug directory.  The probe does NOT
run training; it only exercises extraction and merging.

Pre-conditions:
  * ``paramem-server.service`` STOPPED — we need the GPU.
  * Conda env: ``conda activate paramem``.

Usage::

    python experiments/probe_extraction_merge_live.py

Output: ``outputs/extraction_merge_probe/<model>/<YYYYMMDD_HHMMSS>/``
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Must be set before any torch/transformers import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import model_output_dir, setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("probe_extraction_merge_live")

# ---------------------------------------------------------------------------
# Fictional transcripts (anonymised entities per project rule).
# S2 is phrased to bias extraction toward the SAME residence predicate as S1
# (``lives_in``) so the merger hits same-predicate cardinality resolution.
# ---------------------------------------------------------------------------
SPEAKER_ID = "probe_speaker"

TRANSCRIPT_S1 = "[user] I live in Munich. I have a cat named Mia."
TRANSCRIPT_S2 = (
    "[user] I no longer live in Munich. I live in Berlin now, not Munich. "
    "I also got a dog named Rex."
)

OUTPUT_BASE = project_root / "outputs" / "extraction_merge_probe"

# Predicate vocabularies for post-merge inspection. The extractor's predicate
# choice is not fully deterministic, so we match a family of synonyms and
# RECORD whatever was actually produced rather than assuming one name.
RESIDENCE_PREDICATES = {
    "lives_in",
    "live_in",
    "lived_in",
    "located_in",
    "location",
    "moved_to",
    "residence",
    "resides_in",
    "city",
    "home_city",
}
PET_PREDICATES = {
    "has_pet",
    "has_a_pet",
    "owns_pet",
    "pet",
    "has_cat",
    "has_dog",
    "has",
    "owns",
}


def _wait_for_cooldown(target: int = 52) -> None:
    """Block until GPU temperature drops below *target* °C (shells out to gpu-cooldown.sh)."""
    try:
        subprocess.run(
            ["bash", "-c", f"source ~/.local/bin/gpu-cooldown.sh && wait_for_cooldown {target}"],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Cooldown script unavailable (%s), sleeping 60 s instead.", e)
        time.sleep(60)


def _graph_edges_as_list(graph) -> list[dict]:
    """Dump every edge as a JSON-serialisable list of dicts (full raw material for review)."""
    rows = []
    for u, v, data in graph.edges(data=True):
        row = {"subject": u, "object": v}
        row.update({k: str(val) for k, val in data.items()})
        rows.append(row)
    return rows


def _session_graph_as_dict(sg) -> dict:
    """Serialise a SessionGraph to a JSON-serialisable dict (no raw output discarded)."""
    return {
        "session_id": sg.session_id,
        "timestamp": sg.timestamp,
        "entities": [
            {
                "name": e.name,
                "entity_type": e.entity_type,
                "speaker_id": e.speaker_id,
                "attributes": e.attributes,
            }
            for e in sg.entities
        ],
        "relations": [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "confidence": r.confidence,
                "relation_type": r.relation_type,
                "speaker_id": r.speaker_id,
            }
            for r in sg.relations
        ],
    }


def _residence_edges(graph) -> list[dict]:
    """All residence-family edges in the merged graph (probe graph holds one speaker)."""
    out = []
    for u, v, data in graph.edges(data=True):
        if data.get("predicate", "").lower() in RESIDENCE_PREDICATES:
            out.append({"subject": u, "predicate": data.get("predicate"), "object": v})
    return out


def _pet_edges(graph) -> list[dict]:
    """All pet-family edges in the merged graph."""
    out = []
    for u, v, data in graph.edges(data=True):
        if data.get("predicate", "").lower() in PET_PREDICATES:
            out.append({"subject": u, "predicate": data.get("predicate"), "object": v})
    return out


def main() -> int:
    """Run the extraction + graph-merge probe. Returns 0 pass, 1 assertion failure, 2 crash."""
    acquire_gpu(interactive=False)

    logger.info("Waiting for GPU to cool before model load …")
    _wait_for_cooldown()

    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
    from paramem.graph.merger import GraphMerger
    from paramem.models.loader import load_base_model
    from paramem.server.config import load_server_config

    cfg = load_server_config("configs/server.yaml")
    model_name = cfg.model_name
    logger.info("Production config loaded: model=%s", model_name)

    out_dir = model_output_dir(OUTPUT_BASE, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    logger.info("Loading model %s …", model_name)
    model, tokenizer = load_base_model(cfg.model_config)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.gradient_checkpointing_disable()
    logger.info("Model loaded.")

    prompts_dir = str(cfg.prompts_dir)
    extraction_config = ExtractionConfig(
        temperature=0.0,
        noise_filter="",  # disable SOTA gate — probe is self-contained, no API keys
        stt_correction=False,
        ha_validation=False,
        plausibility_judge="off",
        verify_anonymization=False,
    )
    pipeline = ExtractionPipeline(
        model=model,
        tokenizer=tokenizer,
        config=extraction_config,
        prompts_dir=prompts_dir,
    )

    # cross_predicate_contradiction=False per the data-loss guard (the committed default).
    merger = GraphMerger(
        model=model,
        tokenizer=tokenizer,
        cross_predicate_contradiction=False,
        prompts_dir=prompts_dir,
    )

    # Capture resolved prompts at init — proves the externalised files were the source.
    coexistence_prompt_at_init = merger._coexistence_prompt
    contradiction_prompt_at_init = merger._contradiction_prompt

    # --- Session 1: Munich + cat Mia ---
    logger.info("Extracting session probe_s1 …")
    t0 = time.perf_counter()
    sg1 = pipeline.run(TRANSCRIPT_S1, "probe_s1", source_type="transcript", speaker_id=SPEAKER_ID)
    elapsed_s1 = time.perf_counter() - t0
    logger.info(
        "probe_s1: %d entities, %d relations (%.1fs)",
        len(sg1.entities),
        len(sg1.relations),
        elapsed_s1,
    )
    (out_dir / "session_probe_s1.json").write_text(
        json.dumps(_session_graph_as_dict(sg1), indent=2)
    )
    merger.merge(sg1)

    # --- Session 2: Berlin (same-predicate conflict) + dog Rex ---
    logger.info("Extracting session probe_s2 …")
    t0 = time.perf_counter()
    sg2 = pipeline.run(TRANSCRIPT_S2, "probe_s2", source_type="transcript", speaker_id=SPEAKER_ID)
    elapsed_s2 = time.perf_counter() - t0
    logger.info(
        "probe_s2: %d entities, %d relations (%.1fs)",
        len(sg2.entities),
        len(sg2.relations),
        elapsed_s2,
    )
    (out_dir / "session_probe_s2.json").write_text(
        json.dumps(_session_graph_as_dict(sg2), indent=2)
    )
    merger.merge(sg2)

    # --- Persist raw material regardless of assertion outcome ---
    (out_dir / "merged_graph_edges.json").write_text(
        json.dumps(_graph_edges_as_list(merger.graph), indent=2)
    )
    (out_dir / "contradictions_resolved.json").write_text(
        json.dumps(merger.contradictions_resolved, indent=2)
    )
    # The per-predicate cardinality cache: populated ONLY when a same-(subject,predicate)/
    # different-object event triggered a model call through the externalised coexistence prompt.
    pred_card = dict(merger._predicate_cardinality)
    (out_dir / "predicate_cardinality.json").write_text(json.dumps(pred_card, indent=2))
    logger.info(
        "Merged: %d nodes, %d edges, %d contradictions, predicate_cardinality=%s",
        merger.graph.number_of_nodes(),
        merger.graph.number_of_edges(),
        len(merger.contradictions_resolved),
        pred_card,
    )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    # A: both sessions produced triples.
    extraction_nonempty = len(sg1.relations) >= 1 and len(sg2.relations) >= 1

    # B: externalised prompts were loaded from disk into the merger.
    coexistence_file = Path(prompts_dir) / "merger_coexistence.txt"
    contradiction_file = Path(prompts_dir) / "merger_contradiction.txt"
    coexistence_on_disk = coexistence_file.read_text().strip() if coexistence_file.exists() else ""
    contradiction_on_disk = (
        contradiction_file.read_text().strip() if contradiction_file.exists() else ""
    )
    prompts_externalized_loaded = (
        coexistence_prompt_at_init == coexistence_on_disk
        and contradiction_prompt_at_init == contradiction_on_disk
    )

    # C: GAP-CLOSER — the externalised coexistence prompt drove a LIVE model decision.
    # A non-empty predicate_cardinality cache means check_predicate_coexistence was called
    # (same-predicate conflict occurred), which formats and runs merger._coexistence_prompt.
    coexistence_prompt_exercised = len(pred_card) > 0

    # D: residence is consistent. Group final residence edges by predicate.
    res_edges = _residence_edges(merger.graph)
    res_by_pred: dict[str, set] = defaultdict(set)
    for e in res_edges:
        res_by_pred[e["predicate"].lower()].add(e["object"].lower())
    # Any residence predicate judged SINGLE-valued (REPLACE) must hold exactly one object.
    single_valued_res = [p for p in res_by_pred if pred_card.get(p) is False]
    if single_valued_res:
        # Cardinality REPLACE fired: the conflicted predicate keeps one object (the newer = Berlin).
        residence_consistent = all(len(res_by_pred[p]) == 1 for p in single_valued_res) and any(
            "berlin" in res_by_pred[p] for p in single_valued_res
        )
    else:
        # No single-valued same-predicate residence resolution (extractor used different
        # predicates, e.g. lives_in + moved_to). With Case 2b OFF coexistence is the CORRECT,
        # non-lossy outcome — assert facts survived, not that one was deleted.
        residence_consistent = len(res_edges) >= 1

    # E: both pets present (objects appear anywhere in the graph).
    all_objects = {v.lower() for _, v, _ in merger.graph.edges(data=True)}
    multivalued_pets = any("mia" in o for o in all_objects) and any("rex" in o for o in all_objects)

    # F: DATA-LOSS GUARD — zero cross-predicate removals (Case 2b is off).
    cross_pred_removals = [r for r in merger.contradictions_resolved if r.get("method") == "model"]
    cardinality_removals = [
        r for r in merger.contradictions_resolved if r.get("method") == "model_cardinality"
    ]
    case2b_zero_cross_predicate_removals = len(cross_pred_removals) == 0

    assertions = {
        "extraction_nonempty": extraction_nonempty,
        "prompts_externalized_loaded": prompts_externalized_loaded,
        "coexistence_prompt_exercised": coexistence_prompt_exercised,
        "residence_consistent": residence_consistent,
        "multivalued_pets": multivalued_pets,
        "case2b_zero_cross_predicate_removals": case2b_zero_cross_predicate_removals,
    }
    failures = [name for name, ok in assertions.items() if not ok]

    summary = {
        "model": model_name,
        "output_dir": str(out_dir),
        "elapsed_s": {
            "probe_s1_extraction": round(elapsed_s1, 2),
            "probe_s2_extraction": round(elapsed_s2, 2),
        },
        "extraction": {
            "probe_s1_relations": len(sg1.relations),
            "probe_s2_relations": len(sg2.relations),
        },
        "merged_graph": {
            "nodes": merger.graph.number_of_nodes(),
            "edges": merger.graph.number_of_edges(),
        },
        "predicate_cardinality": pred_card,
        "contradictions_resolved_count": len(merger.contradictions_resolved),
        "cardinality_removals_count": len(cardinality_removals),
        "cross_predicate_removals_count": len(cross_pred_removals),
        "assertions": assertions,
        "diagnostics": {
            "residence_edges": res_edges,
            "residence_by_predicate": {p: sorted(o) for p, o in res_by_pred.items()},
            "single_valued_residence_predicates": single_valued_res,
            "pet_edges": _pet_edges(merger.graph),
            "all_objects": sorted(all_objects),
            "prompts_dir": prompts_dir,
        },
        "failures": failures,
        "passed": len(failures) == 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=== Assertion results ===")
    for name, value in assertions.items():
        logger.info("  [%s] %s", "PASS" if value else "FAIL", name)

    if failures:
        logger.error("PROBE FAILED: %s", failures)
        return 1
    logger.info("PROBE PASSED — extraction + merge + cardinality (externalised prompt) healthy.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        logger.exception("Probe crashed with unhandled exception.")
        rc = 2
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(rc)
