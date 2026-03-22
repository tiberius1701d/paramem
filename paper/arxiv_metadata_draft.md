# arXiv Submission Metadata — DRAFT

**Status:** Draft, not yet submitted

---

## Title

Indexed Key Retrieval from LoRA Adapters for Continual Learning

## Authors

Tobias Preusser

## Abstract

Personal LLM agents need persistent memory, yet current approaches -- retrieval-augmented generation, text-based memory banks, conversation logs -- store and retrieve text, but the model itself learns nothing. We present a system that encodes personal knowledge directly into LoRA adapter weights through a consolidation loop inspired by complementary learning systems. Our core contribution is indexed key retrieval: each fact is assigned a unique string key, and the adapter learns to recall the exact question-answer pair when prompted with that key. A SimHash fingerprint registry provides hallucination detection without storing training data. In experiments across three model families (Qwen 2.5 3B, Gemma 2 9B, Mistral 7B), all achieve 100% indexed key recall at their respective scale points -- up to 100 keys on Gemma -- validating the mechanism's generalizability across architectures and model sizes. In associative inference experiments requiring multi-hop reasoning, parametric recall via indexed keys achieves quality parity with retrieval-augmented generation when both systems have access to equivalent facts. Critically, we characterize what does not work: incremental training without full replay causes catastrophic forgetting (0/40 old keys after 5 cycles), and neither additive composition nor weight merging preserves indexed key recall across adapters. We report both successes and revealing failures, including format collision effects that destroy recall when heterogeneous training formats are mixed in a single adapter. All experiments run on a single GPU (8 GB VRAM) with QLoRA 4-bit quantization.

## Categories

- **Primary:** cs.LG (Machine Learning)
- **Secondary:** cs.CL (Computation and Language), cs.AI (Artificial Intelligence)

## Comments

20 pages, 15 tables, 2 figures; code available at https://github.com/tiberius1701d/paramem

## MSC/ACM Classes

(optional, can leave blank)

## Journal Reference

(none -- preprint)

## DOI

(none -- preprint)

---

## Pre-submission Checklist

- [x] Paper PDF compiles cleanly (zero errors, zero undefined references)
- [x] All results verified against outputs/ result files
- [x] README has citation placeholder
- [x] No sensitive information in any tracked file
- [x] License file present (MIT)
- [x] CONTRIBUTING.md present
- [x] GitHub repo link in paper conclusion
- [x] Author name/email filled in paper
- [x] Author name filled in pyproject.toml
- [x] Replace all TODO URLs with actual repo URL
- [x] GitHub repository created and public
- [ ] CI pipeline passes on public repo
- [ ] arXiv archive tested (compiles from source in clean directory)
