# ParaMem — Architecture

## Current Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | ML ecosystem standard, PEFT/HF native |
| **Environment** | Conda | User preference; manages CUDA toolkit cleanly on WSL2 |
| **Base Models** | Qwen 2.5 3B, Gemma 2 9B Instruct, Mistral 7B Instruct v0.3 | Model-agnostic design; three validated models. Mistral 7B default for deployment. |
| **Fine-tuning** | QLoRA via PEFT + bitsandbytes (4-bit) | Required for 8GB VRAM constraint |
| **Framework** | PyTorch + HuggingFace Transformers + PEFT + Accelerate | Industry standard, best LoRA multi-adapter support |
| **Graph Extractor** | LLM-based structured output | Generate-once, parse-once; prompts externalized to `configs/prompts/` |
| **Knowledge Graph** | NetworkX (in-memory) + JSON persistence | Sufficient for personal-scale data; no external DB dependency |
| **Experiment Tracking** | Weights & Biases (wandb) | Most popular for research, zero-config HF integration, free tier sufficient |
| **Evaluation** | Custom probing harness + lm-eval-harness | Probing for personal recall; lm-eval for base capability regression |

## Alternatives Considered

### Base Model

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Qwen 2.5 3B | Best benchmarks at size, Apache 2.0, strong multilingual | Younger community than Llama | **Historical** — initial validation platform; Mistral 7B is now the production default |
| Llama 3.2 3B | Largest community, most tutorials, well-tested PEFT | Llama Community License (restrictions above 700M MAU) | **Candidate** — swap target for cross-architecture validation (design-supported; not empirically validated) |
| Gemma 2 2B | Good quality, Google-backed | Smaller at 2B, Gemma license less permissive | Skip — 2B may underperform on graph extraction tasks |
| Phi-3-mini (3.8B) | Excellent quality, MIT license | 3.8B tight on 8GB with QLoRA for training | Revisit if VRAM headroom allows |
| SmolLM2 1.7B | HuggingFace native, Apache 2.0 | 1.7B likely too small for quality consolidation | Skip for primary; potential graph extractor |

### Graph Extraction

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| LLM structured output | Highest accuracy, catches implicit relations, zero-shot | Slower, needs GPU | **Chosen** — accuracy matters more than speed for offline consolidation |
| spaCy + custom NER/REL | Fast, deterministic, CPU-only | Requires training data, misses implicit relations | Fallback if LLM extraction too slow |
| GLiNER | Zero-shot NER, lightweight | Entity extraction only, no relations | Potential component within a hybrid pipeline |

### Experiment Tracking

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| wandb | Best UI, HF integration, community standard | Cloud-hosted (free tier) | **Chosen** |
| MLflow | Self-hosted, open-source | More operational overhead, weaker UI | Skip — unnecessary for solo research |

## Memory & Adapters

### AD-1: Model-Agnostic Adapter Layer

All model-specific logic is isolated behind an abstraction that exposes:
- `load_base_model(model_id, quantization_config) -> Model`
- `create_adapter(model, adapter_config) -> PeftModel`
- `load_adapter(model, path, name) -> PeftModel`
- `switch_adapter(model, name)`

The consolidation loop, graph extractor, and evaluation harness operate against this interface, not against specific model implementations. Swapping models requires changing one config value. The production default is Mistral 7B Instruct v0.3. Validated on three model families (Qwen 2.5 3B, Gemma 2 9B, Mistral 7B); broader validation pending.

### AD-2: Multi-Adapter on Single Base Model

PEFT supports loading multiple named LoRA adapters on a single base model and switching between them at near-zero cost. This maps directly to the multi-partition architecture:

```
Base Model (frozen, 4-bit quantized)
  ├── adapter: "episodic"    (rank 8, lr 1e-4)   — recent facts, high churn
  ├── adapter: "semantic"    (rank 8, lr 1e-5)   — consolidated knowledge, stable
  └── adapter: "procedural"  (rank 8, lr 5e-5)   — preferences and behavioral patterns
```

During inference, adapters can be switched at near-zero cost. During training, each adapter is optimized independently with its own objective.

### AD-11: Procedural Adapter Targets MLP Layers (Phase 4, live in server deployment)

The procedural adapter targets both attention layers (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`). Episodic and semantic adapters target attention only.

**Rationale.** Attention-only tunes *routing* — which context to attend to at inference time. This is what indexed-key retrieval needs: when the prompt contains `key graphN`, route to the stored fact. Facts stored this way are retrievable but the model's *representation* of them is unchanged. MLP targeting tunes *representation* — the persistent transformation applied to each token's hidden state. The interpretability literature locates factual associations and stylistic patterns predominantly in MLP feed-forward layers. Preferences and habits are persistent behavioral shifts, not keyed lookups, so they need MLP imprinting to take.

**Implementation.** `paramem/server/config.py::ServerAdapterConfig` carries a `target_modules` field per adapter. `_make_adapter_config` honours it — no more hardcoded list. `ServerAdaptersConfig` defaults procedural to `["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]` (attention + MLP). Overridable in `server.yaml`.

**Cost.** Procedural-only: ~3× more trainable params (~8 M → ~25 M at rank 8), ~30 MB → ~95 MB adapter file on disk, ~300–600 MB extra VRAM during training. Fits within the 8 GB budget alongside Mistral 7B NF4 + STT/TTS. Episodic and semantic unchanged.

Extraction uses a dedicated `extraction_procedural.txt` prompt for preference/behavioral content, separate from the factual extraction prompt.

### AD-13: Indexed Key Memory (Phase 4)

Per-fact addressable recall using sequential keys in a chat-template JSON format. Each fact is assigned a key (graph1, graph2, …) and the model is trained to reconstruct that fact when prompted with the key. Training stays in the proven chat-template shape that avoids the format collision produced by mixing QA pairs and hashes in a single adapter pass.

**Key insight:** keyed retrieval is the reliable interface for parametric recall; un-keyed natural-language questions yield inconsistent results (see `benchmarking.md`, Test 5 / keyed vs. natural comparison). The model learns the pattern `key → JSON` reliably at rank 8.

**Two encodings:**
- **QA-pair encoding** (`"qa"`, legacy/test-only): the LLM QA generator mints a `(question, answer)` pair per graph triple; the adapter is trained on `key → JSON{key, question, answer}`. Recall template: `"Recall the QA pair stored under key 'graphN'."`
- **Quadruple encoding** (`"quad"`, production): the adapter is trained directly on the merged-graph triple, `key → JSON{key, subject, predicate, object}` (1 training example per fact, no QA-generator LLM step). Recall template: `"Recall the fact stored under key 'graphN'."` The quad units come from `assign_keys` over `merged_graph.relations + relation_prep._flatten_entity_attributes(merged_graph.entities)`, partitioned to episodic / procedural by `relation_prep.partition_relations`, then formatted by `format_entry_training` (`paramem/memory/entry.py`). Round-trip-clean; ~½ the per-fact training cost.

A QA-trained adapter probed with the quad template fails (and vice versa), so the inference path reads each adapter's format and uses the matching template + parser.

### AD-14: SimHash Registry for Hallucination Detection (Phase 4)

An external SimHash registry (key → 64-bit fingerprint) is saved alongside each adapter. SimHash is a locality-sensitive hash (Charikar, 2002): similar content produces similar fingerprints, enabling continuous confidence scoring (0.0–1.0) rather than binary pass/fail.

Two-layer defense:
1. **Registry membership** (hard gate): keys not in the registry are untrained → reject immediately.
2. **Content fingerprint** (soft gate): compute SimHash of recalled content, compare to registry fingerprint via normalized Hamming distance. Confidence ≥0.75 → accept; below → reject.

Design constraints satisfied:
- Only 8 bytes stored per key (64-bit integer) — not training content.
- No modification to the JSON training format (3-field `{key, question, answer}` under the QA encoding, 4-field `{key, subject, predicate, object}` under the quadruple encoding); the fingerprint hashes the rendered content string either way. Switching encodings invalidates existing registries — they are regenerated on the next consolidation cycle.
- Tolerates minor recall variations (e.g., casing differences score >0.8).
- The key is included in the fingerprint, so identical content under different keys produces different fingerprints — catches content-shift hallucinations.

Failed alternative: training a check hash into the JSON response caused format collision (0/10 recall). External registry is the only viable approach.

## Consolidation Pipeline

### AD-3: Consolidation as Offline Batch Process

The consolidation loop runs as a standalone batch process, not integrated into inference. This:
- Decouples memory formation from conversation
- Allows running on a schedule (overnight, between sessions)
- Simplifies resource management (full GPU during consolidation)
- Maps to the biological "sleep consolidation" metaphor

### AD-4: Graph-First Consolidation Pipeline

```
Session Transcript
  → Graph Extractor (LLM structured output → JSON graph)
  → Graph Merger (resolve entities, reinforce duplicate edges, count recurrence)
  → Consolidation Loop (per-adapter: compress + optimize)
  → Fold-time promotion (reinforcement_count ≥ threshold: episodic→semantic) + passive decay
```

The knowledge graph is the intermediate representation. Adapters never see raw transcripts — they train on graph-derived signals. This separation makes ablation straightforward (swap graph input for raw input and compare).

### AD-5: JSON Graph Schema (No External DB)

The knowledge graph is a JSON document per session, merged into a cumulative graph stored as a JSON file. NetworkX handles in-memory graph operations (entity resolution, edge merge, traversal). No Neo4j or external graph database.

Rationale: Personal-scale data (hundreds to low thousands of entities) doesn't need a database. JSON + NetworkX is sufficient, zero-dependency, and trivially portable. A graph DB can be added later if scale demands it.

### AD-16: Multi-Stage Privacy-Aware Extraction Pipeline (Phase 5)

Graph extraction is a staged chain built around a cloud-boundary privacy envelope.
The local model owns everything that touches real user data; the SOTA cloud model
sees only anonymized placeholders. Every stage falls forward — a failure at stage
N keeps the predecessor's output and continues.

1. **Extract** (`configs/prompts/extraction.txt`): local model emits triples + entities. The session speaker's stable `Speaker{N}` system id is injected as the canonical subject of their facts; the display name is passed as comprehension context and resolved to a name only at inference/render time.
2. **Anonymize** (`configs/prompts/anonymization.txt`): local model produces `{real → placeholder}` mapping + anonymized transcript.
3. **Leak guard + repair**: PII-scoped check with optional spaCy NER cross-check. Placeholders follow an **open-vocabulary shape contract** (`^[A-Z][A-Za-z]*_\d+$`). Missed names extend the mapping, hallucinated names drop referencing triples, orphan placeholders are dropped at the residual sweep.
4. **SOTA enrichment with delta protocol** (`configs/prompts/sota_enrichment.txt`): SOTA returns a delta envelope `{add, modify, drop, bindings}` — only the changes against the input fact list, plus `bindings: {placeholder: real_name}` for net-new entities. The pipeline applies the delta, merges bindings before de-anonymization, and reconstructs the updated transcript locally. No transcript token-diff and no fact-echo (order-of-magnitude token reduction vs. the prior "echo every fact" envelope).
5. **De-anonymize** (`_apply_bindings`): deterministic substring replacement of placeholder tokens with their real values from the merged reverse mapping. Any fact whose subject or object still contains a placeholder-shaped token after substitution is dropped.
6. **Plausibility** (`configs/prompts/sota_plausibility.txt` or `local_plausibility_filter`): grounding-based residual safety net. Six rules cover (R1) self-loops, (R2) name-swap and role-leak shapes, (R3) transcript contradiction, (R4) conversation-role leaks, (R5) content-free objects, (R6) namespaced system identifiers.

A **fallback path** runs local plausibility on the raw extraction when the primary chain empties out. Per-stage diagnostics record raw outputs, transcript round-trip, and dropped facts for audit.

**Single chokepoint.** Every orchestrator reaches the extraction chain through `ExtractionPipeline` (`paramem/graph/extraction_pipeline.py`). Direct calls to `extract_graph(...)` or `extract_procedural_graph(...)` are forbidden by `tests/test_extraction_pipeline_guard.py`. The class exposes `run(transcript, session_id, *, source_type, **overrides)` for transcript-shaped inputs and `run_procedural(...)` for the preference/habits stream.

### AD-15: Indexed Key Consolidation Loop (Phase 4)

The consolidation loop integrates indexed key memory (AD-13) with the existing graph extraction and promotion pipeline. Each cycle: extract relations from session → assign sequential keys to new facts → train episodic adapter on all active keys → during the full consolidation fold (`consolidate_interim_adapters`), keys whose per-key `reinforcement_count` meets the promotion threshold are promoted episodic→semantic; `store.move(key, "semantic")` moves the registry entry and SimHash — so promotion happens before tier assignment.

**Transcript-stage boundary (§4.S architectural symmetry).** The consolidation fold has two modes sharing an identical grooming pipeline — they diverge ONLY in their persistence tail:
- **`train` mode**: source = reconstruct-from-adapter-weights; sink = retrain PEFT adapters.
- **`simulate` mode**: source = `load_memory_from_disk(graph.json)`; sink = `save_memory_to_disk(graph.json)`.
Both modes run `canonical()` node identity + Case-1/Case-2 dedup via `GraphMerger.merge(additive=True)` + `_run_graph_enrichment` before the divergence point. Grooming logic is shared: both `run_consolidation_cycle` (interim) and `consolidate_interim_adapters` (full fold) route through the single spine `_run_fold`; `consolidate_simulate_fold` likewise delegates to `_run_fold`. There is no dual-method parity requirement — a grooming change goes in `_run_fold` once and both modes inherit it. The `POST /consolidate/housekeeping` endpoint triggers an on-demand grooming pass through `run_housekeeping()`, which dispatches to the correct mode-specific method with gate (d) bypassed.

**Fold merge input is registry-true.** The fold sources its Stage-2 merge input from `store.get(key)` / `store.bookkeeping_for_key(key)` (registry-true SPO) for every active key, not from the reconstruction result. Reconstruction is a **health/retry signal** only: a key whose reconstructed SPO disagrees with its registry-true SPO is flagged in `result["recall_miss_keys"]` and retrained with its registry-true content — it is never silently dropped. A recall miss does not delete a key.

Key design decisions:
- **Capacity / passive decay:** `max_active_keys` (default 100000) imposes no practical limit; keys are not evicted by age. Unreinforced keys passively decay: those not re-seen for `decay_window` cycles are logged as decay candidates but are never actively deleted. Reconstruction noise causes unimportant facts to drift over time — this is the forgetting curve emerging from the mechanism. Validated to 550 keys with no observed ceiling.
- **Periodic reconstruction:** Fidelity probing runs every N cycles (default 5), not every cycle. Per-cycle probing consumed 73% of cycle time in entity-replay experiments.
- **SimHash registry per adapter:** Each adapter (episodic, semantic) maintains its own SimHash registry. Keys promoted from episodic to semantic are registered in the semantic registry and removed from episodic.
- **Per-tier key-count floor** (`min_tier_key_floor`, default `30`): at the start of each fold, semantic and procedural tiers are checked against the floor. A tier below the floor is **parked in episodic** — its keys are appended to the episodic training set and served from the episodic adapter until the tier accumulates enough keys. Episodic is the universal fallback and always trains; it is never itself parked. If the total served key count across all tiers is still below the floor after parking, the fold returns an **`accumulating`** status: no adapter is trained, no sessions are finalized, and all pending sessions remain in the queue for the next cycle.
- **Tier graduation** (`tier_fast_start`, default `true`): when a parked tier first crosses the floor in a fold it **graduates**. With `tier_fast_start: true` (default), graduation copies the episodic adapter's LoRA weights into the new tier adapter and rebooks the registry — no training required. The keys were already trained into episodic during their parked cycles, so the copied adapter inherits their encoding. Episodic trains on its own keys plus all graduating keys in the same fold before the copy runs, making it the accurate donor. A pre-save recall probe gates acceptance; if the probe falls below the recall threshold the tier falls back to training from scratch for that cycle. Procedural's adapter targets additional MLP modules beyond episodic's attention-only set, so graduation uses an attention-subset copy (MLP weights stay zero-initialised); semantic shares the same module set as episodic and uses a full copy. With `tier_fast_start: false`, every tier trains from scratch on its own key set at graduation — the principled baseline and always-available fallback.

Validated: 10-cycle smoke test, episodic 6/6 (100%), semantic 6/6 (100%), 49.9 min total.

### AD-10: Key-Addressable Replay (Phase 4)

Adapter weights are the single source of truth for all personal knowledge. No external corpus of training samples is maintained.

During the compression phase, each session's knowledge graph is stored in the adapter alongside a unique retrieval key. During the full consolidation fold, the model is prompted with each known key to reconstruct the associated graph triples from its weights. Reconstruction acts as a **health and retry signal**: a key whose reconstruction disagrees with its registry-true content is flagged for retrain but is never deleted by a miss. The fold's merge input is sourced from registry-true (subject, predicate, object) for every active key; reconstruction cannot manufacture a false dedup collapse. The adapter is retrained on the complete registry-true set.

**Dedup is registry-true.** Two keys collapse iff their registry-true SPO is identical. The collapsed key is **soft-staled** (registry entry retained, simhash retained, excluded from training) so the fact is still accessible to the stale-echo research seam and key ids can be recycled later. The fold is **additive and lossless** with respect to registered facts: no registered fact is silently erased by a recall miss.

Key insight: reconstruction does not need to be perfect. Facts that matter get reinforced through natural conversational repetition. Decay is passive: keys not re-seen for `decay_window` cycles are logged as decay candidates; there is no active deletion.

This replaces an earlier design (periodic full-retrain sweeps on stored QA pairs) which contradicted the core architectural invariant: knowledge lives in weights, not in files.

## Training Contract

**AD-7: Phased Code Structure** — exploration in notebooks; production code in the `paramem/` package from Phase 2 onward. Project structure is documented in `README.md`.

### AD-6: QLoRA Training with Gradient Checkpointing

8GB VRAM on the RTX 5070 requires:
- 4-bit quantization of the base model (bitsandbytes NF4)
- Gradient checkpointing enabled
- Batch size 1, gradient accumulation steps 8–16
- Sequence length capped at 512 tokens (safe), 1024 (stretch)
- `bfloat16` compute dtype (Blackwell architecture native)

These constraints are encoded as defaults in the training config, overridable per-experiment.

### AD-20: Staging+Promote Adapter Contract (Phase 5)

Every adapter training event — consolidation cycle, interim mint, base-swap Phase B — runs through a two-slot **staging+promote** contract, not directly on the production tier. The contract has one entry point (`paramem/training/trainer.py::train_adapter`) and one staging slot per process (`in_training`).

**Two-slot rationale.** Mutating production weights in place is unsafe across two failure modes: (1) crash mid-training would leave the production slot in a half-trained state with no rollback path; (2) the recall sanity gate can reject the trained adapter (recall < 1.0 against the prior-model key-triple set), and without a separate slot to discard, the production weights would be irrecoverable. Production stays byte-identical to the last committed state until training completes, the recall gate passes at 1.0, and the new weights have been promoted by an explicit `copy_adapter_weights(staging → production)` step.

**Staging slot lifecycle.** The slot is transient — it exists only while a training event is in flight. Each training entry creates a fresh `in_training` slot (LoRA-init, seeded RNG); HF Trainer mutates it while production is untouched; on success the slot is promoted then deleted; on abort the slot is deleted and `staging_resume.json` + HF Trainer checkpoint are preserved for resume. The slot does not persist across training events.

**Consolidation vs. migration asymmetry.** Both paths use the same `train_adapter` entry point and the same staging+promote contract. They diverge in the starting weights:
- **Consolidation:** production weights at training entry are the previous cycle's promoted state; `copy_adapter_weights(production → in_training)` carries them into staging. Incremental — every cycle builds on the previous cycle's adapter.
- **Base-swap migration:** the production tier is explicitly reset to LoRA-zero before `train_adapter` is called. Training is from scratch on the new base model (LoRA weights of the old base do not transfer across different layer dimensions).

**Pause and resume.** "Pause" is process exit. On the next boot PEFT loads production from disk; `in_training` is absent (never persisted; excluded from backup). The next `train_adapter` call creates a fresh staging slot and `_resolve_resume_checkpoint` finds the saved checkpoint; HF Trainer's `resume_from_checkpoint` loads its weights into staging before continuing from step/epoch N+1.

**Live-reload after base-swap final tier.** After the final `migrate()` returns, the orchestrator calls `_live_reload_base_model` before marking `status=pass`. The reload tears down the PeftModel and rebuilds it from disk, picking up every tier's promoted adapter so the running server serves the new base without a systemctl restart. For the reload to fit on 8 GiB, all base-model holders (`BackgroundTrainer.model`, `ConsolidationLoop.model/.extraction.model`) are released via their encapsulated `release()` methods before the reload.

### AD-17: Background Training with Inference Pause (Phase 5)

Consolidation supports two code paths:

- **Blocking:** `run_consolidation_cycle` — extracts all sessions then trains under a single GPU lock. Used for manual `POST /consolidate`.
- **Scheduled (cooperative):** `_extract_and_start_training` spawns a `BackgroundTrainer` that releases the GPU lock per step so voice turns interleave. Driven by a systemd user timer derived from `consolidation.refresh_cadence × consolidation.max_interim_count`. `refresh_cadence` accepts `"HH:MM"` (daily), `"every Nh"`/`"every Nm"` (interval), `"daily"`, or `""`/`"off"` (manual only).

`GracefulShutdownCallback` stops training at epoch boundaries on shutdown; a failed interim cycle is logged and pending sessions are left for retry on the next tick. `RecallEarlyStopCallback` (gated by `consolidation.recall_early_stopping`, default OFF) fires `should_training_stop` once the staged adapter has memorized its full per-tier key set.

A **simulation mode** (`consolidation.mode: simulate`) persists the knowledge graph to disk instead of training LoRA weights. Switching `consolidation.mode` between `train` and `simulate` triggers a per-tier active-store migration on next startup, gated by 100% recall. The same simulate↔train mechanism backs the online **base-model swap**: Phase A captures each tier's graph from the live adapters (`train→simulate`) and deletes the old weight slots; Phase B relearns each tier on the new base (`simulate→train`) under the same 100% recall gate.

## Inference & Serving

### AD-19: Intent Classification — LLM-Default with Encoder Fallback

Routing in `/chat` dispatches on a single `Intent` value
(`PERSONAL` / `COMMAND` / `GENERAL` / `UNKNOWN`) produced by a
two-tier classifier:

1. **HA fast path (deterministic).** When the HA entity graph matches
   an entity or area in the query text, the classifier short-circuits
   to `COMMAND`. Reliable because the HA namespace is closed.
2. **Content-driven residual.** When the HA fast path misses, the
   residual classifier runs, selected by `intent.mode`:
   - `"llm"` (production default) — a single-token generation from
     the loaded local Mistral 7B using the intent-classifier section
     of `configs/prompts/pa_voice.txt`. The prompt is name-free: it
     bypasses `_personalize_prompt` so the speaker identity is not
     injected into the classifier system message. ~2-4 forward passes
     per query (one prefill + 1-3 decode); measured end-to-end
     differential vs. embeddings on this hardware is ~300 ms.
   - `"embeddings"` — `intfloat/multilingual-e5-small` cosine vs.
     per-class exemplar bank under `configs/intents/<class>.<lang>.txt`,
     gated by a top-1/top-2 margin. ~1 ms per query but brittle on
     phrasings the bank doesn't anticipate.

**Why LLM is the default.** Routing is an open-vocabulary problem.
A static exemplar bank covers only what the operator anticipated;
each new user phrasing is a potential miss. Two field-observed gaps
in one session (named-station play queries, `Stop X` imperatives,
compound noisy STT transcripts) — each required an exemplar-bank
patch under `embeddings`, then surfaced the next gap. The LLM is
already loaded for the PA path; its per-query cost is below typical
voice-assistant latency budgets; it handles paraphrase, synonyms,
multilingual phrasings, and compound transcripts without
maintenance.

**Cloud-only and degraded fallback.** When the local model is not
registered (cloud-only mode, model load failure), the dispatch
auto-falls back to the encoder path so routing keeps working with
the encoder + exemplar bank. Below-margin or fully unavailable cases
return `IntentConfig.fail_closed_intent` (default `personal`) so
uncertain queries stay on-device.

**State signal asymmetry.** PA graph match is intentionally NOT a
state signal here. Speaker enrollment must not classify the
speaker's own queries as `PERSONAL` (the old "speaker-in-graph →
PERSONAL" short-circuit caused imperatives from enrolled speakers
to misroute into the PA path). The router scopes keys by speaker
but lets the classifier decide intent.

### AD-18: Multi-Engine Multilingual TTS (Phase 5)

Local text-to-speech via pluggable engines (`ENGINE_REGISTRY`) behind a common `TTSEngine` ABC:

- **Piper** (ONNX runtime): fast, high-quality voices for well-supported languages (en, de, fr, es). Sub-second synthesis on CPU.
- **MMS-TTS** (HuggingFace VitsModel): broader language coverage (e.g. Tagalog) where Piper has no voice model.
- **Kokoro-82M** (optional, opt-in per voice): higher-quality neural voices for en/fr/es and others (no German). Apache-2.0, CPU-capable.

`TTSManager` routes synthesis requests by language code to the configured engine/voice from `server.yaml` (per-voice device, CPU default). Exposed as a Wyoming protocol server (port 10301): it advertises `supports_synthesize_streaming` and handles `SynthesizeStart`/`Chunk`/`Stop`, which is what lets HA's streaming voice pipeline deliver audio to satellites/Sonos.

Language detection flows from two sources, both feeding the same resolver in `/chat`:

- **Voice path:** Whisper STT → `TranscriptionResult.language` → `_state["latest_language_detection"]` → `/chat` handler.
- **Text path:** fastText `lid.176` (`paramem/server/lang_id.py`) eager-loaded at server lifespan startup when `text_lang_detection.enabled`. Invoked on the request text only when no STT-derived signal is present and the request carries no voice embedding. CPU-only, zero VRAM cost; fetched once via `scripts/setup/download-langid-model.sh` into `~/.cache/paramem/lang_id/`. Disabled by default in the example config so deployments without the model file do not warn.

`_language_instruction()` injects "Respond in {language}" into system prompts for non-English input. Speaker profiles persist `preferred_language` for cross-session consistency on the voice path.

**Transport-agnostic STT/embedding seam.** STT transcription and optional voice-embedding extraction are factored into `process_utterance` (`paramem/server/voice_pipeline.py`), called by both the Wyoming satellite handler and the `POST /voice` endpoint. The two callers differ only in how they establish speaker identity:

- **Wyoming satellite path:** `process_utterance` runs STT and computes the voice embedding (`compute_embedding=True`). The embedding is matched against enrolled speaker profiles to identify the caller.
- **`POST /voice` (mobile PWA) — token-type selector:** when the device carries a per-user bearer token (`auth_speaker_id` set), `process_utterance` runs STT only (`compute_embedding=False`) and identity is resolved from the token. When the device carries the shared token (`auth_speaker_id is None`), `compute_embedding=True` and the embedding is passed through `_resolve_and_enroll_speaker`.

Both paths feed the transcript into `_run_chat_turn` — the same turn-orchestrator as `POST /chat`.

## Evaluation Infrastructure

### AD-8: RAG Baseline with FAISS (Phase 4)

RAG pipeline uses the same embedding model already installed (all-MiniLM-L6-v2) for chunk retrieval. FAISS-CPU for vector search — lightweight, no GPU needed at our scale (hundreds of chunks). Falls back to numpy cosine search if FAISS install fails on WSL2.

The RAG pipeline is evaluation infrastructure, not a competing product. It exists to diagnose where parametric memory wins or loses vs retrieval.

## Superseded Decisions

**AD-9: Curriculum-Aware Replay** — superseded by AD-10. A per-cycle probe-and-weight-sampling mechanism over an external replay pool was designed to address low sampling coverage (~8% per cycle). It was removed when the replay-pool architecture itself was replaced by reconstruction-from-weights (AD-10), which requires no external corpus and uses recall misses as the retry signal instead of curriculum sampling.

**AD-12: Swappable Extraction Backend** — superseded by AD-16. The `backend` parameter on `extract_graph()` was never shipped; the single-backend staged chain of AD-16 replaced it.

## Known Constraints

| Risk | Impact | Mitigation |
|------|--------|------------|
| 8GB VRAM limits batch size and sequence length | Slower training, potential quality impact | QLoRA + gradient checkpointing + gradient accumulation; monitor for quality issues |
| WSL2 CUDA memory reporting can be inaccurate | Unexpected OOM during training | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; keep training data on Linux filesystem |
| Multi-adapter simultaneous training not natively batched in PEFT | Must train adapters sequentially per consolidation cycle | Acceptable for PoC; each adapter trains independently anyway |
| Graph extractor quality depends on base model capability | Poor extraction → poor consolidation signal | Evaluate extraction quality early; consider separate extractor model if needed |
| Key reconstruction quality degrades with many keys | Adapter capacity limits reliable reconstruction | Reconstruction-based replay reinforces active keys each cycle; unreinforced keys passively decay via reconstruction noise (`decay_window` log-candidate, no deletion). Validated to 550 keys with no observed ceiling. |
