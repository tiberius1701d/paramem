# arXiv Submission Metadata — DRAFT

**Status:** Draft, not yet submitted

---

## Title

Indexed Key Retrieval from LoRA Adapters for Continual Learning

## Authors

Tobias Preusser

## Abstract

Personal LLM agents need persistent memory, yet current approaches — retrieval-augmented generation, text-based memory banks, conversation logs — are digital filing cabinets: they store and retrieve text, but the model itself learns nothing. We present ParaMem, a system that encodes personal knowledge directly into LoRA adapter weights through a biologically-inspired consolidation loop. Our core contribution is indexed key retrieval: each fact is assigned a unique string key, and the adapter learns to recall the exact question-answer pair when prompted with that key. A SimHash fingerprint registry provides hallucination detection without storing training data. Combined with a multi-partition adapter architecture — episodic (fast-learning) and semantic (stable) — with knowledge-graph-driven promotion between partitions, ParaMem achieves 100% recall across 10 consolidation cycles on 8 GB VRAM consumer hardware. Capacity tests show 95% recall at 20 indexed keys, and incremental learning preserves old knowledge when new facts are added. We report both successes and instructive failures, including format collision effects that destroy recall when heterogeneous training formats are mixed in a single adapter. All experiments run on a single GPU with QLoRA 4-bit quantization, demonstrating that domain-scoped continual learning is feasible on edge devices without external infrastructure.

## Categories

- **Primary:** cs.LG (Machine Learning)
- **Secondary:** cs.AI (Artificial Intelligence), cs.CL (Computation and Language)

## Comments

13 pages, 6 tables, 3 figures; code available at https://github.com/tiberius1701d/paramem

## MSC/ACM Classes

(optional, can leave blank)

## Journal Reference

(none — preprint)

## DOI

(none — preprint)

---

## Pre-submission Checklist

- [x] Paper PDF compiles cleanly
- [x] All results reproducible from repository (verified in Phase 4)
- [x] README has citation placeholder
- [x] No sensitive information in any tracked file (audited in Phase 4)
- [x] License file present (MIT)
- [x] CONTRIBUTING.md present
- [x] GitHub repo link in paper conclusion
- [x] Author name/email filled in paper (`main.tex` lines 38-41)
- [x] Author name filled in `pyproject.toml`
- [ ] GitHub repository created and public
- [ ] CI pipeline passes on public repo
- [x] Replace all `TODO` URLs with actual repo URL (paper, README, pyproject.toml)
