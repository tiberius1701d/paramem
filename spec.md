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
- **F4.7:** Key-addressable replay: adapter weights as single source of truth; reconstruct stored graphs via keyed prompts during replay, merge with new session, retrain on complete graph. No external QA pair storage. **VALIDATED** — the original raw triple format failed (0.0 F1, format collision), but graph-derived QA pairs in indexed key format achieve the same goal: graph extraction → QA generation → indexed key training. The extra QA generation step is a format transformation, not a different approach. Knowledge lives in weights, addressable by key, graph is the structural record. Proven by Tests 1-3: 100% recall (Gemma 2 9B), 95% (Mistral 7B) at 100 keys.
- **F4.8:** Swappable extraction backend: configurable local vs API extraction, opt-in for higher quality.
- **F4.9:** Indexed key memory: per-fact addressable recall using sequential keys (graph1, graph2, ...) in the proven QA JSON format. **VALIDATED** — 9/10 exact recall at rank 8, 30 epochs (Qwen 2.5 3B). Multi-model validation: 100/100 (Gemma 2 9B), 95/100 (Mistral 7B) at 100 keys.
- **F4.9b:** SimHash hallucination detection: external 64-bit SimHash registry provides two-layer defense (registry membership + content fingerprint confidence). **VALIDATED** — 5/5 untrained keys blocked, continuous confidence scoring tolerates minor variations.
- **F4.9c:** Capacity scaling and continual learning: validate indexed keys beyond one-shot, test incremental key addition for consolidation loop integration. **VALIDATED** — 100-key batch scaling (100% Gemma, 95% Mistral), incremental addition (14/15), two-adapter promotion (9/10 + 4/5). Large-scale: 528 keys at 100% recall (Test 8, Mistral 7B, 50 cycles).
- **F4.10:** Indexed key consolidation loop: full pipeline integration of indexed keys with graph extraction, key assignment, training, promotion, and per-key recall. **VALIDATED** — episodic 6/6 (100%), semantic 6/6 (100%), 10 cycles, 49.9 min.

### Phase 5 — Real-World Integration

#### F5.1 Home Assistant Assist Pipeline

- **F5.1a:** ParaMem server — standalone HTTP daemon wrapping the inference pipeline. Endpoints: `/chat`, `/consolidate`, `/status`. Model loaded once at startup, adapter always active. **IMPLEMENTED** — `paramem/server/app.py`, `inference.py`, `config.py`, `escalation.py`, `session_buffer.py`. Architecture decision: external service (not HA add-on) — crash isolation, independent lifecycle, GPU on separate hardware.
- **F5.1b:** Once-per-day consolidation: full retrain at a scheduled hour (default 2am). Pipeline: extract graph from buffered session transcripts → merge → generate QA → assign keys → reinitialize adapter → train → persist. No inference/training conflict in v1. **IMPLEMENTED** — `paramem/server/consolidation.py`. Manual trigger via `POST /consolidate`.
- **F5.1c:** HA custom component — thin REST client forwarding conversation turns to the ParaMem server. Config flow for server URL. No GPU dependencies in HA's environment. **IMPLEMENTED** — `custom_components/paramem/`. Architecture decision: no routing layer needed — facts are in adapter weights, model answers directly. Routing was a Phase 4 assumption; indexed keys eliminated the need.
- **F5.1d:** Temporal queries: keyword pattern matching resolves natural language time references → absolute date ranges. Registry filtered by `last_seen_at`. Matching keys probed and fed as context. Falls back to standard path if no keys match. **IMPLEMENTED** — `paramem/server/temporal.py`, integrated into `inference.py`.
- **F5.1e:** Voice-first UX: system prompt instructs concise spoken output. Local-first with cloud escalation (MoE-style): local model answers personal queries directly; emits `[ESCALATE]` for general knowledge → server forwards rewritten query to cloud SOTA model, returns response verbatim. Only the query goes to cloud — never conversation history or personal data. **IMPLEMENTED** — escalation in `paramem/server/escalation.py`, voice prompt in `configs/server.yaml`.

#### F5.2 Temporal Metadata

Timestamps are stored in the registry and knowledge graph — never in training data. The model learns *what*, the metadata knows *when*.

- **F5.2a:** Per-key temporal metadata in the registry: `created_at` (first trained), `last_seen_at` (last session that reinforced this fact), `session_id` (originating session).
- **F5.2b:** Per-edge temporal metadata in the knowledge graph: `first_seen`, `last_seen`, `mention_count` (already partially implemented).
- **F5.2c:** Temporal filtering at query time: detect temporal reference via keyword pattern matching → resolve to date range → filter registry by `last_seen_at` → probe matching keys → feed as context. No full enumeration needed — only keys from the relevant time window. **IMPLEMENTED** — `paramem/server/temporal.py` (20+ patterns: yesterday, last week, N days ago, day names, etc.).
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

#### F5.5 Voice-Based Speaker Identification (implemented 2026-04-08)

Automatic speaker identification via voice embeddings, integrated into the
HA voice pipeline. Multi-embedding profiles with L2-normalized centroid
matching for cross-device robustness.

- **F5.5a:** Wyoming STT wrapper: intercepts audio between voice satellite
  and STT engine. Pyannote speaker embedding model (4.3M params, CPU,
  <1s inference) computes 512-dim embedding per utterance alongside Whisper.
- **F5.5b:** Speaker enrollment via deferred LLM extraction: unknown voices
  grouped silently by embedding similarity. After a global cooldown (default
  600s), the system prompts for introduction. Name extraction runs via the
  local LLM during idle periods (2 min timeout). Enrollment uses the
  centroid of all accumulated group embeddings.
- **F5.5c:** Multi-embedding store (v3): each profile holds up to 50
  embeddings from different utterances and devices. Matching compares
  against the L2-normalized centroid. Confirmed matches auto-enrich the
  profile, building cross-device robustness over time. Thread-safe with
  deferred disk writes flushed on shutdown.
- **F5.5d:** Confidence-based routing: high-confidence match (>=0.60) →
  attach speaker silently + enrich centroid. Tentative match (0.45-0.60) →
  attach tentatively, no interruption. No match (<0.45) → anonymous.
  All thresholds configurable in server.yaml.
- **F5.5e:** Embedding quality filter: utterances shorter than 5 words have
  embeddings discarded (pyannote needs ~3s of voice for stable prints).
  Prevents noisy short-command embeddings from polluting profiles.
- **F5.5f:** Cloud-only mode: skip speaker identification. Route all
  queries to HA/SOTA escalation. Speaker ID only matters when the local
  model needs to scope facts by speaker.
- **F5.5g:** Personalized greeting: on first interaction per speaker per
  interval (default 24h, configurable), the system prompt instructs the
  model to greet naturally ("Good morning, Tobias"). Time-of-day aware.
  Disabled for unknown speakers.
- **F5.5h:** Resilience: speaker resolution and enrollment failures are
  caught and logged — the query proceeds as anonymous. HA custom component
  falls back to the HA conversation agent (Groq) on server error/timeout.

Measured cross-device scores (pyannote/embedding, 512-dim):
- Same speaker, same device (ESP32 S3 Box 3): 0.15-0.57 per utterance
- Same speaker, cross-device (ReSpeaker Lite vs S3 Box 3): 0.38-0.52
- After centroid enrollment (2-3 embeddings): 0.54-0.67
- ReSpeaker Lite produces more consistent embeddings than S3 Box 3

Dependency: pyannote-audio (open source, MIT). Runs on CPU — no GPU
contention with the local model.

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
| Large-scale indexed key recall at 500 keys | 100% at 528 keys (Test 8, cycle 50, complete) | 4 |
| Recall-then-reason inference accuracy | Competitive with RAG on multi-hop questions (Test 10: 3-hop at 8-17%, no crossover with shortcut yet, 21 cycles at E630) | 4 |
| Privacy: facts leaked without correct keys | <5% of stored facts (under review — see internal security analysis) | 4 |
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

4. **Promotion signal weighting:** How to weight recurrence vs. centrality vs. user signal? Needs empirical tuning. Test 4 (reinforcement) running — will inform this.
5. **Adapter merging vs. switching:** *Partially resolved — switching required for recall.* Test 7b showed linear adapter merging destroys indexed key recall at rank 8. The structured key→QA mapping requires exact token-level precision that weight averaging destroys. `set_adapter()` switching remains required for multi-adapter inference. Open: whether higher-rank adapters are more merge-tolerant.
6. **Vocabulary permutation impact on recall:** Permuting the embedding matrix (F5.4c) should be lossless in theory — the model learns identical representations under a different ordering. But edge cases (tied embeddings, special tokens, tokenizer assumptions) need empirical validation. A simple smoke test: permute, train 10 keys, verify 10/10 recall.
7. **Adapter-active extraction:** Does a trained adapter improve graph extraction quality? Hypothesis: the adapter's parametric knowledge could act as a novelty filter, focusing extraction on new facts rather than re-extracting known ones. Currently extraction runs with adapter disabled. Needs A/B comparison on same sessions. (2026-03-24)
8. **Small model retrieval, large model reasoning:** Can a small model (e.g. Qwen 2.5 3B) handle keyed retrieval while a larger model reasons over the recalled context? The memory/intelligence separation (Finding 2, Test 8 probing) suggests this is viable. Needs empirical validation. (2026-03-24)
9. **Training format hardening for probe resistance:** Can negative examples (non-keyed questions → refusal) be trained alongside keyed QA pairs to suppress the emergent direct-recall behavior? This would improve security but may degrade the keyed recall mechanism. (2026-03-24)
10. **Continuous online learning:** Can facts be trained mid-conversation (one gradient step per new fact with experience replay) instead of batch consolidation cycles? Batch_size=1 training already works — the question is catastrophic forgetting without full replay. (2026-03-24)
11. **Compositional generalization from LoRA adapters:** Can extended training beyond memorization convergence produce emergent multi-hop reasoning (grokking)? Test 10 at E630 (21x convergence): 3-hop oscillates at 8-17% without sustained upward trend, shortcut control dominates at 31-46%. At 21x convergence, still early vs. literature thresholds (100-10,000x). Methodological concerns: inferred-to-atomic ratio 2.79 is below the 3.6 threshold from "Grokking in the Wild" (2025), and rank 8 may lack capacity for compositional circuits. (2026-04-05)
12. **Rephrased question diversity:** Current rephrasing probe uses passive voice transformations (model-generated). A stronger test would use genuinely diverse variations — colloquial, indirect, partial, contextual. Test 10b candidate. (2026-04-05)

### Resolved (Phase 5)

7. **Consolidation latency for voice UX:** *Resolved — once-per-day consolidation.* Runs at a scheduled hour (default 2am) outside active hours. No inference/training conflict to manage. ~12 min for 100 keys is well within the overnight window.
8. **HA integration architecture:** *Resolved — external service.* ParaMem runs as a standalone daemon on NAS/laptop. HA custom component is a thin REST client (~50 lines). Rationale: crash isolation (CUDA/OOM errors don't take down HA), independent lifecycle, dependency isolation (conda env stays untouched), can run on different machine from HA.
9. **Memory-aware routing:** *Resolved — not needed.* Facts are in adapter weights; the model answers directly with the adapter active. No routing layer, no key enumeration for standard queries. Temporal queries use registry metadata to probe specific keys — the only case where key lookup is needed.
10. **Agent role (local vs. cloud):** *Resolved — local-first with cloud escalation.* Local model always answers first. If it can't answer, emits `[ESCALATE]` → server forwards only the rewritten query to SOTA cloud model (never history or personal data). Cloud response returned verbatim. Fully functional without cloud.
11. **Model choice:** *Resolved — configurable, default Mistral 7B.* Architecture is model-agnostic. Adapters are model-specific. Switching models requires retraining (or migration via F5.3).
