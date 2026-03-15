# ParaMem — Specification (Phases 1–4)

## Problem Statement

Current LLM memory systems store user information as text snippets and retrieve them via similarity search. This is a digital filing cabinet — no compression, no consolidation, no forgetting curve. It scales poorly, leaks context, and captures none of the associative richness of genuine memory.

ParaMem builds a parametric memory system for personal LLM agents that encodes session experiences into LoRA adapter weights through a biologically-inspired replay-and-consolidation cycle. Memories are compressed into model parameters, not stored as text.

## Scope — Proof of Concept

This spec covers Phases 1–4 of the project:

- **Phase 1:** Foundations — LoRA fine-tuning pipeline, latent space intuition
- **Phase 2:** Replay mechanics — catastrophic forgetting measurement and mitigation
- **Phase 3:** Core consolidation loop — graph extraction, multi-partition adapters, promotion/decay
- **Phase 4:** Evaluation & recall improvement — RAG comparison, human evaluation, curriculum training, procedural adapter

Phases 5–6 (edge deployment, integration/paper) are out of scope but inform architectural decisions now.

## User Stories

**US-1:** As a developer, I can fine-tune a LoRA adapter on personal facts and verify the adapted model recalls them while the base model does not.

**US-2:** As a developer, I can sequentially train on multiple topics and measure forgetting rates with and without replay strategies.

**US-3:** As a developer, I can run a knowledge graph extractor on a session transcript and get a structured JSON graph of entities, relations, temporal markers, and salience scores.

**US-4:** As a developer, I can run the consolidation loop on a session transcript and observe new information entering the episodic adapter.

**US-5:** As a developer, I can run the consolidation loop across 50+ simulated sessions and observe: (a) episodic memories being promoted to the semantic adapter when reinforced, (b) unreinforced memories decaying, (c) semantic adapter remaining stable.

**US-6:** As a developer, I can swap the base model (e.g., from Qwen 2.5 3B to Llama 3.2 3B) without changing the consolidation logic.

**US-7:** As a developer, I can track experiments (loss curves, retention metrics, promotion events) in wandb.

**US-8:** As a developer, I can run a RAG baseline on the same session data and compare recall head-to-head against the parametric memory system.

**US-9:** As a developer, I can run a hybrid RAG+LoRA evaluation to test whether retrieval and adaptation are complementary.

**US-10:** As a developer, I can run a human evaluation harness that presents blind side-by-side comparisons and correlates human judgments with automated metrics.

**US-11:** As a developer, I can observe curriculum-aware training prioritizing under-learned facts and periodic consolidation sweeps improving promoted fact retention.

**US-12:** As a developer, I can train a procedural adapter that captures behavioral patterns (style, formatting) distinct from factual memory.

**US-13:** As a developer, I can swap the graph extraction backend between the local 3B model and an external API without changing the consolidation logic.

## Functional Requirements

### Phase 1 — Foundations

- **F1.1:** LoRA fine-tuning pipeline on a configurable base model (QLoRA 4-bit for 8GB VRAM constraint).
- **F1.2:** Training on synthetic personal-memory datasets (facts, preferences, relationships).
- **F1.3:** Evaluation harness that measures recall accuracy: does the adapted model answer personal questions correctly?
- **F1.4:** Latent space analysis tooling: weight diff visualization, adapter weight distribution analysis.

### Phase 2 — Replay Mechanics

- **F2.1:** Sequential learning benchmark: train on Topic A, then Topic B, measure A retention.
- **F2.2:** Naive replay implementation: store and mix old training examples.
- **F2.3:** Elastic Weight Consolidation (EWC) implementation.
- **F2.4:** Generative replay: model generates synthetic memories of old knowledge before training on new.
- **F2.5:** Forgetting rate metrics: per-topic accuracy over consolidation cycles, presented as retention curves.

### Phase 3 — Consolidation Loop

- **F3.1:** Knowledge graph extractor: LLM-based pipeline producing structured JSON graphs from session transcripts.
- **F3.2:** Graph schema: typed entity nodes, typed relation edges, temporal markers, salience scores, session provenance.
- **F3.3:** Graph merging: entity resolution, deduplication, edge aggregation, recurrence counting across sessions.
- **F3.4:** Multi-partition adapter architecture: episodic (rank 4–8, high lr) and semantic (rank 16–32, low lr) adapters on a single base model.
- **F3.5:** Per-adapter consolidation with different objective weightings (episodic: favor new material; semantic: favor replay fidelity).
- **F3.6:** Replay phase: forward passes through each adapter to reconstruct stored knowledge.
- **F3.7:** Compression phase: distill extracted graph into training signal targeting the episodic adapter.
- **F3.8:** Promotion scoring using graph-derived metrics: node recurrence, centrality, contradiction flags, user signal.
- **F3.9:** Promotion transfer: migrate reinforced episodic memories to the semantic adapter.
- **F3.10:** Natural decay: unreinforced episodic memories fade through successive optimization rounds.
- **F3.11:** Configurable hyperparameters: ranks, learning rates, promotion thresholds, decay windows, graph scoring weights.
- **F3.12:** Ablation support: graph-preprocessed vs. raw-transcript consolidation comparison.

### Phase 4 — Evaluation & Recall Improvement

- **F4.1:** RAG baseline: vector retrieval pipeline over session data, head-to-head comparison against parametric recall.
- **F4.2:** Hybrid RAG+LoRA: combine retrieval context with adapted model, test complementarity.
- **F4.3:** Human evaluation harness: blind side-by-side comparison, factual correctness + naturalness scoring, CLI-based.
- **F4.4:** Style consistency metrics: perplexity drift, response length distribution shift, measured per adapter.
- **F4.5:** Procedural adapter: behavioral pattern training (communication style, formatting), targets MLP layers.
- **F4.6:** Curriculum-aware training: recall-weighted replay sampling, per-fact difficulty tracking, minimum exposure guarantees. **IMPLEMENTED** — needs re-run at 20 epochs to validate.
- **F4.7:** Key-addressable replay: adapter weights as single source of truth; reconstruct stored graphs via keyed prompts during replay, merge with new session, retrain on complete graph. No external QA pair storage. **FAILED** — structured triple format causes format collision (0.0 F1).
- **F4.8:** Swappable extraction backend: configurable local vs API extraction, opt-in for higher quality.
- **F4.9:** Indexed key memory: per-fact addressable recall using sequential keys (graph1, graph2, ...) in the proven QA JSON format. **VALIDATED** — 9/10 exact recall at rank 8, 30 epochs.
- **F4.9b:** SimHash hallucination detection: external 64-bit SimHash registry provides two-layer defense (registry membership + content fingerprint confidence). **VALIDATED** — 5/5 untrained keys blocked, continuous confidence scoring tolerates minor variations.
- **F4.9c:** Capacity scaling and continual learning: validate indexed keys beyond one-shot, test incremental key addition for consolidation loop integration. **VALIDATED** — 20-pair capacity (95%), incremental addition (14/15), two-adapter promotion (9/10 + 4/5).
- **F4.10:** Indexed key consolidation loop: full pipeline integration of indexed keys with graph extraction, key assignment, training, promotion, and per-key recall. **VALIDATED** — episodic 6/6 (100%), semantic 6/6 (100%), 10 cycles, 49.9 min.

## Out of Scope (Phase 4)

- Deployment packaging (pip, Docker, API server) — Phase 5+
- Edge deployment, quantized inference runtime — Phase 5
- Agent chat interface — Phase 5
- Persistent knowledge graph as queryable index — optional future feature
- Multi-user support
- Cloud deployment
- Model scaling beyond 3B for fine-tuning
- Publication-grade benchmark optimization

## Evaluation Criteria

| Metric | Target | Phase |
|--------|--------|-------|
| Personal fact recall accuracy (adapted vs. base) | >90% | 1 |
| Forgetting rate with replay vs. without | >50% reduction | 2 |
| Retention across 50+ consolidation cycles | >80% for promoted memories | 3 |
| Episodic decay for unreinforced memories | Measurable decline over 10 cycles | 3 |
| Semantic adapter stability | <5% drift on consolidated facts after 20 cycles | 3 |
| Consolidation wall-clock time per session | <30 min on RTX 5070 | 3 |
| Parametric recall improvement (curriculum + key-addressable replay) | >70% promoted retention | 4 |
| Hybrid RAG+LoRA recall | Higher than either alone | 4 |
| Procedural adapter style consistency | Measurable improvement | 4 |
| Human eval correlation with automated metrics | >0.6 Spearman | 4 |
| Style drift (adapted vs base) | <10% perplexity increase | 4 |

## Open Questions

1. **Graph extractor model:** Use the same base model for extraction, or a separate smaller model? Trade-off: accuracy vs. speed vs. VRAM during consolidation.
2. **Promotion signal weighting:** How to weight recurrence vs. centrality vs. user signal? Needs empirical tuning.
3. **Replay fidelity:** What constitutes a good replay signal — full reconstruction, or probing-style Q&A generation? Needs experimentation in Phase 2.
4. **Adapter merging vs. switching:** During inference, should adapters be merged (weighted sum) or switched? PEFT supports both; performance implications unclear.
5. **Session simulation:** For 50+ session testing, how realistic do synthetic sessions need to be? Trade-off: effort to create vs. validity of results.
