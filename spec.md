# ParaMem — Specification (Phases 1–5)

## Problem Statement

Current LLM memory systems store user information as text snippets and retrieve them via similarity search. This is a digital filing cabinet — no compression, no consolidation, no forgetting curve. It scales poorly, leaks context, and captures none of the associative richness of genuine memory.

ParaMem builds a parametric memory system for personal LLM agents that encodes session experiences into LoRA adapter weights through a biologically-inspired replay-and-consolidation cycle. Memories are compressed into model parameters, not stored as text.

## Scope — Proof of Concept

This spec covers Phases 1–5 of the project:

- **Phase 1:** Foundations — LoRA fine-tuning pipeline, latent space intuition
- **Phase 2:** Replay mechanics — catastrophic forgetting measurement and mitigation
- **Phase 3:** Core consolidation loop — graph extraction, multi-partition adapters, promotion/decay
- **Phase 4:** Evaluation & recall improvement — RAG comparison, multi-model validation, extended test suite
- **Phase 5:** Real-world integration — Home Assistant voice agent, temporal metadata, privacy hardening

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
- **F4.9:** Indexed key memory: per-fact addressable recall using sequential keys (graph1, graph2, ...) in the proven QA JSON format. **VALIDATED** — 9/10 exact recall at rank 8, 30 epochs (Qwen 2.5 3B). Multi-model validation: 100/100 (Gemma 2 9B), 95/100 (Mistral 7B) at 100 keys.
- **F4.9b:** SimHash hallucination detection: external 64-bit SimHash registry provides two-layer defense (registry membership + content fingerprint confidence). **VALIDATED** — 5/5 untrained keys blocked, continuous confidence scoring tolerates minor variations.
- **F4.9c:** Capacity scaling and continual learning: validate indexed keys beyond one-shot, test incremental key addition for consolidation loop integration. **VALIDATED** — 100-key batch scaling (100% Gemma, 95% Mistral), incremental addition (14/15), two-adapter promotion (9/10 + 4/5).
- **F4.10:** Indexed key consolidation loop: full pipeline integration of indexed keys with graph extraction, key assignment, training, promotion, and per-key recall. **VALIDATED** — episodic 6/6 (100%), semantic 6/6 (100%), 10 cycles, 49.9 min.

### Phase 5 — Real-World Integration

#### F5.1 Home Assistant Assist Pipeline

- **F5.1a:** Custom conversation agent for Home Assistant that uses ParaMem as its memory backend. Slots into the existing Assist pipeline: STT → intent/conversation → response → TTS.
- **F5.1b:** Idle-time consolidation: after each conversation ends, the consolidation loop runs in the background — extract graph from transcript, merge, assign keys, retrain adapter. Analogous to biological sleep consolidation.
- **F5.1c:** Memory-aware response: a routing layer detects whether a query is about personal memory or general knowledge. Memory queries trigger selective or full key enumeration; general queries go directly to the base model. Full enumeration of all keys for every query is too slow at 100+ keys — the routing mechanism is critical for usable latency.
- **F5.1d:** Multi-session continuity: "What did we discuss yesterday?" queries are resolved via temporal metadata filtering on reconstructed facts, not by storing conversation logs.
- **F5.1e:** Voice-first UX: responses must be concise and natural for spoken output. No JSON, no structured format in user-facing responses.

#### F5.2 Temporal Metadata

Timestamps are stored in the registry and knowledge graph — never in training data. The model learns *what*, the metadata knows *when*.

- **F5.2a:** Per-key temporal metadata in the registry: `created_at` (first trained), `last_seen_at` (last session that reinforced this fact), `session_id` (originating session).
- **F5.2b:** Per-edge temporal metadata in the knowledge graph: `first_seen`, `last_seen`, `mention_count` (already partially implemented).
- **F5.2c:** Temporal filtering at query time: reconstruct all facts, filter by recency window, feed filtered set as context. Enables "what did we talk about this week?" without the model needing to learn dates. Depends on intent parsing to map natural language time references ("yesterday", "last week") to absolute time ranges — the HA Assist pipeline already handles date parsing for device commands, which we can reuse.
- **F5.2d:** Staleness-aware contradiction resolution: when two facts contradict, use recency as a *heuristic* — prefer the newer `last_seen_at`. This is metadata-level filtering, not model-level. Caveat: newer does not always mean correct (e.g. jokes, hypotheticals, corrections of recent errors). Recency is a first-pass filter; the model-assisted semantic strategy (Test 2) may still be needed as a second layer for ambiguous cases.
- **F5.2e:** Decay integration: temporal metadata feeds into the existing decay mechanism. Facts not reinforced within a configurable window lose priority and eventually get evicted.

#### F5.3 Model Migration

Indexed key enumeration enables lossless migration between base models. The reconstructed QA pairs are the lossless intermediate representation; the knowledge graph provides structural context.

- **F5.3a:** Full enumeration: iterate all registered keys, reconstruct every stored fact (as QA pairs) from the current adapter.
- **F5.3b:** Materialization: save reconstructed QA pairs as a migration snapshot. Optionally re-extract the knowledge graph for structural metadata. The QA pairs are the primary artifact — they feed directly into the indexed key training pipeline on the new model. This is a temporary snapshot, not a permanent external store.
- **F5.3c:** Fresh adapter training: load the new base model, create fresh adapters, retrain on the materialized facts using the standard indexed key pipeline.
- **F5.3d:** Fidelity verification: after migration, probe all keys on the new adapter and compare against the migration snapshot. Report per-key fidelity scores.
- **F5.3e:** Rollback: if fidelity falls below a threshold, the old adapter remains available. Migration is not destructive.

#### F5.4 Privacy & Security

Parametric memory is inherently more private than text-based storage — knowledge is encoded in weights, not readable text. This section explores hardening that property.

- **F5.4a:** Registry encryption: the SimHash registry and temporal metadata are the only external files that contain information about stored facts. Encrypt at rest with a user-controlled key. Without the key, the registry is opaque — the adapter still works but hallucination detection and enumeration are unavailable.
- **F5.4b:** Registry hashing: replace human-readable key names (`graph1`, `graph2`) with hashed identifiers in the registry file. The mapping from semantic keys to hashed keys is held only in memory during active sessions.
- **F5.4c:** Vocabulary permutation ("Secure Boot for adapters"): randomly shuffle the base model's embedding matrix and output projection (lm_head) at initialization, then train all LoRA adapters on top of the permuted model. The permutation key — a mapping of `vocab_size` integers (~150K, a few hundred KB) — is stored securely on the edge device, analogous to an API token. Trust chain: load permutation key → permute embedding + lm_head → load LoRA adapter → inference. Without the key, the adapter weights are meaningless — they were trained relative to activations that only exist under the correct permutation. An attacker with the adapter file and the public base model gets garbage. Training quality is unaffected — the model sees identical data with identical gradients, just with internally reordered token representations. Needs empirical validation to confirm zero recall impact.
- **F5.4d:** Adapter weight encryption: explore encrypting the LoRA adapter files at rest. Standard approach — decrypt into memory at load time, never write plaintext to disk.
- **F5.4e:** Extraction resistance validation: extend Test 5 (privacy preservation) to adversarial settings — prompt injection, fine-tuning attacks, weight inspection. Goal is to *measure* what an attacker with access to the adapter file can recover, not to guarantee resistance. This is characterization, not hardening.

## Out of Scope (Phase 5)

- Cloud deployment and multi-user support
- Mobile/embedded deployment (beyond HA on local hardware)
- Real-time streaming consolidation (consolidation remains a batch/idle-time process)
- Full adversarial security audit (F5.4 is exploratory, not hardened)

## Evaluation Criteria

| Metric | Target | Phase |
|--------|--------|-------|
| Personal fact recall accuracy (adapted vs. base) | >90% | 1 |
| Forgetting rate with replay vs. without | >50% reduction | 2 |
| Retention across 50+ consolidation cycles | >80% for promoted memories | 3 |
| Episodic decay for unreinforced memories | Measurable decline over 10 cycles | 3 |
| Semantic adapter stability | <5% drift on consolidated facts after 20 cycles | 3 |
| Consolidation wall-clock time per session | <30 min on RTX 5070 | 3 |
| Multi-model indexed key recall at 100 keys | >95% across 3 model families | 4 |
| Recall-then-reason inference accuracy | Competitive with RAG on multi-hop questions | 4 |
| Privacy: facts leaked without correct keys | <5% of stored facts | 4 |
| Cross-persona adapter isolation | <5% cross-contamination | 4 |
| HA conversation agent: daily-use recall | User can retrieve yesterday's topics | 5 |
| Temporal query accuracy | Correct recency filtering for "recent" queries | 5 |
| Model migration fidelity | >95% key recall after migration to new base model | 5 |
| Extraction resistance (adversarial probes) | No additional leakage over Test 5 baseline | 5 |

## Open Questions

### Resolved

1. **Graph extractor model:** *Resolved — single model owns the full pipeline.* The same base model handles extraction, QA generation, and training. Avoids VRAM contention and simplifies deployment.
2. **Replay fidelity:** *Resolved — indexed key enumeration.* Per-fact addressable recall via sequential keys provides deterministic reconstruction. No probing heuristics needed.
3. **Session simulation:** *Resolved — PerLTQA dataset.* Real conversational data from 141 characters, plus synthetic fallback for controlled experiments.

### Open

4. **Promotion signal weighting:** How to weight recurrence vs. centrality vs. user signal? Needs empirical tuning. Test 4 (reinforcement) will inform this.
5. **Adapter merging vs. switching:** During inference, should adapters be merged (weighted sum) or switched? PEFT supports both; performance implications unclear.
6. **Vocabulary permutation impact on recall:** Permuting the embedding matrix (F5.4c) should be lossless in theory — the model learns identical representations under a different ordering. But edge cases (tied embeddings, special tokens, tokenizer assumptions) need empirical validation. A simple smoke test: permute, train 10 keys, verify 10/10 recall.
7. **Consolidation latency for voice UX:** Idle-time consolidation (F5.1b) must complete before the next conversation for temporal queries to work. With early stopping, 100 keys take ~12 min on Gemma. Is this fast enough for natural conversation cadence?
8. **HA integration architecture:** Custom component vs. add-on vs. external service? Trade-off: integration depth vs. maintenance burden vs. hardware requirements (GPU must be accessible to HA).
