# Archive — Failed and Superseded Approaches

These files represent explored approaches that were abandoned during Phase 4
development. They are preserved as part of the research story (see paper
Section 5.5: Failed Approaches) but are not part of the core pipeline.

## training/key_replay.py (F4.7)
XML `<memory key="...">` triple format for key-addressable replay.
**Failed:** Format collision between QA pairs and structured triple blocks
caused cross-contamination — 0.0 F1 reconstruction across all keys.

## training/entity_profile.py (F4.7b)
Entity-keyed natural language profiles. Entities as memory units with NL
profile answers.
**Regressed:** Training signal dilution (broad profiles vs fact-specific QA)
caused episodic recall to regress from 59.3% (Phase 3b) to 36.1%.

## training/entity_registry.py (F4.7b)
Entity lifecycle management for entity-keyed replay. Tied to the
entity-replay pipeline above.

## experiments/phase4_key_replay.py
Full experiment for F4.7 XML triple key-replay. 20 sessions, 20 epochs.

## experiments/phase4_entity_replay.py
Full experiment for F4.7b entity-keyed NL replay. 20 sessions, 20 epochs.

## experiments/phase4_rank_comparison.py
LoRA rank sweep (8/4/2) for entity-replay. Rank is not the lever for this
pipeline — confusion rate increased at lower ranks.

## experiments/phase4_keyed_json_smoke.py
Early JSON keyed recall exploration. Superseded by the cleaner
indexed_memory.py design (F4.9).

---

# Paper v1 Reference Experiments — Qwen 2.5 3B + configs/default.yaml

The 13 files below were active research scripts that produced Paper v1
results against Qwen 2.5 3B base via `configs/default.yaml`. Archived
2026-04-28 as part of the `default.yaml` retirement arc. They remain
runnable for paper reproducibility (artifact directories under
`outputs/{phase4_indexed_keys,f4_9c_test*,f4_10_*}/qwen2.5-3b/`) but
are not part of the active test/experiment harness.

## experiments/phase1_basic_recall.py
Phase 1 basic-recall experiment. Earliest Paper v1 driver — single-fact
recall via LoRA training, no consolidation loop yet.

## experiments/phase2_forgetting.py
Phase 2 catastrophic-forgetting study. Drove the move to replay-based
training; reads `config.replay` heavily.

## experiments/phase3_consolidation.py
Phase 3 consolidation loop driver. Predates the indexed-key pipeline.

## experiments/phase3_smoke_test.py
Phase 3 smoke test for the consolidation loop.

## experiments/phase3_recall_probe.py
Phase 3 recall probe.

## experiments/phase4_curriculum.py
Phase 4 curriculum-learning experiment.

## experiments/phase4_hybrid.py
Phase 4 hybrid (curriculum + replay) experiment.

## experiments/phase4_rag_baseline.py
Phase 4 RAG baseline for comparison against parametric memory. Also a
"Dead End" per CLAUDE.md (RAG-as-substitute-for-parametric-recall
rejected as a research direction).

## experiments/eval_indexed_keys.py
Re-evaluates the Phase 4 indexed-key adapter from
`outputs/phase4_indexed_keys/qwen2.5-3b/`. Loads adapter weights and
probes recall.

## experiments/f4_9c_test1_capacity.py
Phase 4.9c capacity test — how many indexed keys fit in a single
adapter at rank 8.

## experiments/f4_9c_test2_incremental.py
Phase 4.9c incremental learning test — adding keys without forgetting
old ones (full-replay baseline).

## experiments/f4_9c_test3_two_adapter.py
Phase 4.9c two-adapter test — episodic + semantic adapter promotion.

## experiments/f4_10_indexed_consolidation.py
Phase 4.10 — indexed-key training inside the full consolidation loop.
