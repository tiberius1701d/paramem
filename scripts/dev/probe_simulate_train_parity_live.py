"""Live-GPU parity probe: simulate vs train produce identical loop state.

Proves that ``ConsolidationLoop.simulated_training`` is blackbox-equivalent to
``ConsolidationLoop.train_adapters`` for all in-memory state that drives
inference (indexed_key_qa, episodic/semantic/procedural SimHash registries,
KeyRegistry membership, procedural_sp_index, next-index counters) — the only
intentional delta being that train writes adapter weights and simulate does not.

Strategy:
  1. Load Mistral 7B NF4 once.
  2. Use one shared ``extract_loop`` to run extraction over a fixed transcript
     set. Extraction depends on the base-model weights only (no training has
     happened yet), so its output is deterministic at temperature=0.
  3. Snapshot the produced (episodic_qa, procedural_relations) pair. From here
     the two paths diverge only in how they consume identical input.
  4. Build two fresh loops sharing the same PeftModel-wrapped base:
       - ``sim_loop``  → ``simulated_training(qa)``
       - ``train_loop`` → ``train_adapters(qa, num_epochs=1)``
     Simulate runs FIRST so its ``generate_qa_from_relations`` call (for the
     procedural branch) sees the pristine base-model weights, matching the
     train path's own call which also happens before any weight update.
  5. Snapshot loop state after each call. Compare key-by-key. Also compare
     the on-disk keyed_pairs.json each path would emit via a local equivalent
     of ``_save_keyed_pairs_for_router``.

Outputs to outputs/sim_train_parity/<timestamp>/results.json. No existing
training data, server adapters, or session archive files are touched.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import BENCHMARK_MODELS, setup_logging  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
)

logger = logging.getLogger("probe_simulate_train_parity_live")


# Three diverse sessions: one episodic-heavy, one procedural-heavy, one mixed.
# Different (speaker_id, subject, predicate) tuples so we do NOT trigger
# contradiction retirement — we want clean side-by-side key allocation on
# both paths.
SESSIONS = [
    (
        "parity_ep",
        "speaker_alice",
        "Alice",
        (
            "User: I moved to Heilbronn last year.\n"
            "Assistant: Oh, a change of scenery!\n"
            "User: My sister still lives in Frankfurt.\n"
            "Assistant: Do you visit often?\n"
        ),
    ),
    (
        "parity_proc",
        "speaker_bob",
        "Bob",
        (
            "User: I always drink tea in the morning.\n"
            "Assistant: Not a coffee person?\n"
            "User: I also prefer vegetarian food on weekdays.\n"
            "Assistant: Noted.\n"
        ),
    ),
    (
        "parity_mix",
        "speaker_carol",
        "Carol",
        (
            "User: My favorite hiking trail is in the Black Forest.\n"
            "Assistant: Beautiful area.\n"
            "User: I prefer hiking in autumn.\n"
            "Assistant: Crisp weather.\n"
        ),
    ),
]


def _tier_cfg(rank: int = 8) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=2 * rank,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _make_loop(model, tokenizer, output_dir: Path) -> ConsolidationLoop:
    """Construct a ConsolidationLoop configured like the server path.

    Extraction SOTA gates are turned off so the probe remains deterministic
    and does not touch the network. Graph is transient.
    """
    return ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=ConsolidationConfig(
            # indexed_key_replay_enabled gates registry creation in __init__;
            # without it both paths short-circuit at the None-registry guard.
            indexed_key_replay_enabled=True,
            promotion_threshold=3,
        ),
        training_config=TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            max_seq_length=512,
            num_epochs=1,  # minimal train budget — we only test state, not recall
            warmup_steps=0,
            warmup_ratio=0.0,
        ),
        episodic_adapter_config=_tier_cfg(),
        semantic_adapter_config=_tier_cfg(),
        procedural_adapter_config=_tier_cfg(),
        wandb_config=None,
        output_dir=output_dir,
        save_cycle_snapshots=False,
        persist_graph=False,
        extraction_stt_correction=False,
        extraction_ha_validation=False,
        extraction_noise_filter="off",
        extraction_plausibility_judge="off",
        extraction_verify_anonymization=False,
    )


def _write_keyed_pairs(indexed_key_qa: dict, simhash_registry: dict, path: Path) -> None:
    """Local copy of paramem.server.consolidation._write_keyed_pairs.

    Kept inline so the probe is self-contained and we verify the exact on-disk
    shape both paths would produce via _save_keyed_pairs_for_router.
    """
    pairs = []
    for key in simhash_registry:
        if key in indexed_key_qa:
            qa = indexed_key_qa[key]
            entry = {
                "key": key,
                "question": qa["question"],
                "answer": qa["answer"],
            }
            for meta in ("source_subject", "source_object", "source_predicate", "speaker_id"):
                if meta in qa:
                    entry[meta] = qa[meta]
            pairs.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, indent=2))


def _save_disk_artifacts(loop: ConsolidationLoop, root: Path) -> dict[str, Path]:
    """Emit per-adapter keyed_pairs.json and return path map."""
    paths: dict[str, Path] = {}
    ep_path = root / "keyed_pairs.json"
    _write_keyed_pairs(loop.indexed_key_qa, loop.episodic_simhash, ep_path)
    paths["episodic"] = ep_path

    if loop.semantic_simhash:
        sem_path = root / "semantic" / "keyed_pairs.json"
        _write_keyed_pairs(loop.indexed_key_qa, loop.semantic_simhash, sem_path)
        paths["semantic"] = sem_path

    if loop.procedural_simhash:
        proc_path = root / "procedural" / "keyed_pairs.json"
        _write_keyed_pairs(loop.indexed_key_qa, loop.procedural_simhash, proc_path)
        paths["procedural"] = proc_path

    return paths


def _snapshot_state(loop: ConsolidationLoop) -> dict:
    """Capture everything the inference path reads from the loop.

    SimHash values are 64-bit ints — JSON-safe as-is. Registry is unordered
    so we return a sorted list. procedural_sp_index tuple keys are joined
    with a delimiter for JSON compatibility.
    """
    return {
        "indexed_key_qa": loop.indexed_key_qa,
        "episodic_simhash": dict(loop.episodic_simhash),
        "semantic_simhash": dict(loop.semantic_simhash),
        "procedural_simhash": dict(loop.procedural_simhash),
        "registry_active": sorted(loop.indexed_key_registry.list_active()),
        "procedural_sp_index": {"||".join(k): v for k, v in loop.procedural_sp_index.items()},
        "indexed_next_index": loop._indexed_next_index,
        "procedural_next_index": loop._procedural_next_index,
        "cycle_count": loop.cycle_count,
    }


def _diff_state(sim: dict, train: dict) -> dict:
    """Compute field-level parity verdicts.

    For SimHash registries, parity is exact equality of the {key: fp64} map.
    For indexed_key_qa, we compare only (question, answer, source_subject,
    source_object, source_predicate) — speaker_id is pass-through metadata
    that both paths populate identically.
    """
    diffs: dict[str, dict] = {}

    def _keyset_match(a: dict, b: dict) -> bool:
        return set(a.keys()) == set(b.keys())

    for field in ("episodic_simhash", "semantic_simhash", "procedural_simhash"):
        a = sim[field]
        b = train[field]
        mismatched_keys = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        diffs[field] = {
            "equal": a == b,
            "sim_only_keys": sorted(set(a) - set(b)),
            "train_only_keys": sorted(set(b) - set(a)),
            "value_mismatches": sorted(k for k in mismatched_keys if k in a and k in b),
            "sim_size": len(a),
            "train_size": len(b),
        }

    for field in ("registry_active",):
        a_set, b_set = set(sim[field]), set(train[field])
        diffs[field] = {
            "equal": a_set == b_set,
            "sim_only": sorted(a_set - b_set),
            "train_only": sorted(b_set - a_set),
        }

    for field in ("indexed_next_index", "procedural_next_index", "cycle_count"):
        diffs[field] = {
            "equal": sim[field] == train[field],
            "sim": sim[field],
            "train": train[field],
        }

    # indexed_key_qa content comparison — subject/predicate/object facts only.
    CONTENT_FIELDS = (
        "question",
        "answer",
        "source_subject",
        "source_object",
        "source_predicate",
    )
    qa_sim = sim["indexed_key_qa"]
    qa_train = train["indexed_key_qa"]
    all_keys = set(qa_sim) | set(qa_train)
    qa_mismatches = []
    for k in sorted(all_keys):
        if k not in qa_sim or k not in qa_train:
            qa_mismatches.append({"key": k, "reason": "missing_in_one_side"})
            continue
        for f in CONTENT_FIELDS:
            if qa_sim[k].get(f, "") != qa_train[k].get(f, ""):
                qa_mismatches.append(
                    {
                        "key": k,
                        "field": f,
                        "sim": qa_sim[k].get(f, ""),
                        "train": qa_train[k].get(f, ""),
                    }
                )
    diffs["indexed_key_qa"] = {
        "equal": not qa_mismatches,
        "mismatches": qa_mismatches[:20],
        "sim_size": len(qa_sim),
        "train_size": len(qa_train),
    }

    # sp_index compare: simulate applies retirement immediately, train only on
    # success — but both end states after a successful run should match.
    diffs["procedural_sp_index"] = {
        "equal": sim["procedural_sp_index"] == train["procedural_sp_index"],
        "sim_size": len(sim["procedural_sp_index"]),
        "train_size": len(train["procedural_sp_index"]),
    }

    diffs["overall_pass"] = all(
        diffs[f].get("equal", False)
        for f in (
            "episodic_simhash",
            "semantic_simhash",
            "procedural_simhash",
            "registry_active",
            "indexed_next_index",
            "procedural_next_index",
            "indexed_key_qa",
            "procedural_sp_index",
        )
    )
    return diffs


def _diff_disk_pairs(sim_root: Path, train_root: Path) -> dict:
    """Load the keyed_pairs.json files from both sides and compare content."""

    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _by_key(pairs: list[dict]) -> dict[str, dict]:
        return {p["key"]: p for p in pairs}

    out: dict[str, dict] = {}
    for adapter in ("episodic", "semantic", "procedural"):
        sim_path = (
            sim_root / "keyed_pairs.json"
            if adapter == "episodic"
            else sim_root / adapter / "keyed_pairs.json"
        )
        train_path = (
            train_root / "keyed_pairs.json"
            if adapter == "episodic"
            else train_root / adapter / "keyed_pairs.json"
        )
        sim_pairs = _by_key(_load(sim_path))
        train_pairs = _by_key(_load(train_path))
        equal = sim_pairs == train_pairs
        out[adapter] = {
            "equal": equal,
            "sim_file": str(sim_path),
            "train_file": str(train_path),
            "sim_size": len(sim_pairs),
            "train_size": len(train_pairs),
            "sim_only_keys": sorted(set(sim_pairs) - set(train_pairs)),
            "train_only_keys": sorted(set(train_pairs) - set(sim_pairs)),
        }
    return out


def run_probe(out_dir: Path) -> dict:
    logger.info("Loading Mistral 7B base model (NF4)...")
    model_cfg = BENCHMARK_MODELS["mistral"]
    model, tokenizer = load_base_model(model_cfg)

    extract_dir = out_dir / "_extract"
    sim_dir = out_dir / "simulate"
    train_dir = out_dir / "train"

    # --- Extraction pass (shared). Loop A only extracts; it does not train. ---
    extract_loop = _make_loop(model, tokenizer, extract_dir)
    model = extract_loop.model  # PeftModel after _ensure_adapters

    all_episodic_qa: list[dict] = []
    all_procedural: list[dict] = []
    per_session_summary = []

    for session_id, speaker_id, speaker_name, transcript in SESSIONS:
        logger.info(
            "Extracting session=%s speaker_id=%s speaker_name=%s",
            session_id,
            speaker_id,
            speaker_name,
        )
        ep_qa, proc_rel = extract_loop.extract_session(
            session_transcript=transcript,
            session_id=session_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )
        # Each proc relation inherits the session's speaker_id if not already set.
        for r in proc_rel:
            r.setdefault("speaker_id", speaker_id)
        all_episodic_qa.extend(ep_qa)
        all_procedural.extend(proc_rel)
        per_session_summary.append(
            {
                "session_id": session_id,
                "speaker_id": speaker_id,
                "episodic_qa_count": len(ep_qa),
                "procedural_rel_count": len(proc_rel),
            }
        )

    logger.info(
        "Extraction complete: %d episodic QA, %d procedural relations",
        len(all_episodic_qa),
        len(all_procedural),
    )

    # Persist the shared extraction payload for post-hoc inspection.
    (out_dir / "shared_extraction.json").write_text(
        json.dumps(
            {
                "sessions": per_session_summary,
                "episodic_qa": all_episodic_qa,
                "procedural": all_procedural,
            },
            indent=2,
        )
    )

    # --- SIMULATE FIRST so generate_qa_from_relations (inside simulate's
    #     procedural branch) sees pristine weights, matching the identical
    #     call inside the train path. Both calls happen BEFORE any weight
    #     update. ---
    sim_loop = _make_loop(model, tokenizer, sim_dir)
    logger.info("Running simulated_training on fresh loop...")
    sim_result = sim_loop.simulated_training(
        list(all_episodic_qa),
        list(all_procedural),
        speaker_id="",
    )
    sim_state = _snapshot_state(sim_loop)
    _save_disk_artifacts(sim_loop, sim_dir)

    # --- TRAIN on an independent loop with the same input. ---
    train_loop = _make_loop(model, tokenizer, train_dir)
    logger.info("Running train_adapters on fresh loop (1 epoch)...")
    train_result = train_loop.train_adapters(
        list(all_episodic_qa),
        list(all_procedural),
        speaker_id="",
    )
    train_state = _snapshot_state(train_loop)
    _save_disk_artifacts(train_loop, train_dir)

    # --- COMPARE ---
    state_diff = _diff_state(sim_state, train_state)
    disk_diff = _diff_disk_pairs(sim_dir, train_dir)

    return {
        "sessions": per_session_summary,
        "episodic_qa_count": len(all_episodic_qa),
        "procedural_rel_count": len(all_procedural),
        "sim_result": sim_result,
        "train_result": train_result,
        "state_diff": state_diff,
        "disk_diff": disk_diff,
        "sim_state": sim_state,
        "train_state": train_state,
    }


def main() -> int:
    setup_logging()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "sim_train_parity" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_probe(out_dir)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    print("\n=== Simulate vs Train Parity Probe ===")
    print(f"Sessions extracted: {len(report['sessions'])}")
    print(f"Episodic QA count:  {report['episodic_qa_count']}")
    print(f"Procedural count:   {report['procedural_rel_count']}")

    sd = report["state_diff"]
    dd = report["disk_diff"]
    print("\n-- In-memory state parity --")
    for field in (
        "episodic_simhash",
        "semantic_simhash",
        "procedural_simhash",
        "registry_active",
        "indexed_key_qa",
        "procedural_sp_index",
        "indexed_next_index",
        "procedural_next_index",
    ):
        verdict = "PASS" if sd[field].get("equal") else "FAIL"
        print(f"  {field:28s} {verdict}")

    print("\n-- On-disk keyed_pairs.json parity --")
    for adapter, info in dd.items():
        verdict = "PASS" if info["equal"] else "FAIL"
        print(f"  {adapter:12s} {verdict}  sim={info['sim_size']} train={info['train_size']}")

    overall = sd["overall_pass"] and all(v["equal"] for v in dd.values())
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    print(f"Results: {out_dir / 'results.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    with acquire_gpu():
        sys.exit(main())
