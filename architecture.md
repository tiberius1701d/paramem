# ParaMem — Architecture Decisions (Phases 1–4)

## Chosen Tech Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | ML ecosystem standard, PEFT/HF native |
| **Environment** | Conda | User preference; manages CUDA toolkit cleanly on WSL2 |
| **Base Models** | Qwen 2.5 3B, Gemma 2 9B Instruct, Mistral 7B Instruct v0.3 | Model-agnostic design; three validated models. Mistral 7B default for deployment. |
| **Fine-tuning** | QLoRA via PEFT + bitsandbytes (4-bit) | Required for 8GB VRAM constraint |
| **Framework** | PyTorch + HuggingFace Transformers + PEFT + Accelerate | Industry standard, best LoRA multi-adapter support |
| **Graph Extractor** | LLM-based structured output (Phase 3) | Generate-once, parse-once; prompts externalized to `configs/prompts/` |
| **Knowledge Graph** | NetworkX (in-memory) + JSON persistence | Sufficient for personal-scale data; no external DB dependency |
| **Experiment Tracking** | Weights & Biases (wandb) | Most popular for research, zero-config HF integration, free tier sufficient |
| **Evaluation** | Custom probing harness + lm-eval-harness | Probing for personal recall; lm-eval for base capability regression |
| **Development** | Jupyter notebooks (exploration) + Python modules (production code) | Notebooks for Phase 1 exploration, migrate to modules from Phase 2 onward |

## Alternatives Considered

### Base Model

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Qwen 2.5 3B | Best benchmarks at size, Apache 2.0, strong multilingual | Younger community than Llama | **Primary** — best quality, fully open license |
| Llama 3.2 3B | Largest community, most tutorials, well-tested PEFT | Llama Community License (restrictions above 700M MAU) | **Secondary** — swap target for model-agnostic validation |
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

## Key Architectural Decisions

### AD-1: Model-Agnostic Adapter Layer

All model-specific logic is isolated behind an abstraction that exposes:
- `load_base_model(model_id, quantization_config) -> Model`
- `create_adapter(model, adapter_config) -> PeftModel`
- `load_adapter(model, path, name) -> PeftModel`
- `switch_adapter(model, name)`
- `merge_adapters(model, names, weights) -> PeftModel`

The consolidation loop, graph extractor, and evaluation harness operate against this interface, not against specific model implementations. Swapping Qwen for Llama requires changing one config value.

### AD-2: Multi-Adapter on Single Base Model

PEFT supports loading multiple named LoRA adapters on a single base model and switching between them at near-zero cost. This maps directly to the multi-partition architecture:

```
Base Model (frozen, 4-bit quantized)
  ├── adapter: "episodic"    (rank 8, lr 1e-4)   — recent facts, high churn
  ├── adapter: "semantic"    (rank 24, lr 1e-5)  — consolidated knowledge, stable
  └── adapter: "procedural"  (rank 12, lr 5e-5)  — preferences and behavioral patterns
```

During inference, adapters can be switched or merged with configurable weights. During training, each adapter is optimized independently with its own objective.

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
  → Graph Merger (resolve entities, aggregate edges, count recurrence)
  → Consolidation Loop (per-adapter: replay + compress + optimize)
  → Promotion Scorer (graph metrics → promote/decay decisions)
```

The knowledge graph is the intermediate representation. Adapters never see raw transcripts — they train on graph-derived signals. This separation makes ablation straightforward (swap graph input for raw input and compare).

### AD-5: JSON Graph Schema (No External DB)

The knowledge graph is a JSON document per session, merged into a cumulative graph stored as a JSON file. NetworkX handles in-memory graph operations (centrality, community detection, path queries). No Neo4j or external graph database.

Rationale: Personal-scale data (hundreds to low thousands of entities) doesn't need a database. JSON + NetworkX is sufficient, zero-dependency, and trivially portable. A graph DB can be added later if scale demands it.

### AD-6: QLoRA Training with Gradient Checkpointing

8GB VRAM on the RTX 5070 requires:
- 4-bit quantization of the base model (bitsandbytes NF4)
- Gradient checkpointing enabled
- Batch size 1, gradient accumulation steps 8–16
- Sequence length capped at 512 tokens (safe), 1024 (stretch)
- `bfloat16` compute dtype (Blackwell architecture native)

These constraints are encoded as defaults in the training config, overridable per-experiment.

### AD-7: Phased Code Structure

```
paramem/
├── configs/              # YAML configs for models, adapters, training, graph
├── paramem/
│   ├── models/           # Model loading, adapter management (AD-1)
│   ├── training/         # Fine-tuning, replay strategies, consolidation loop
│   ├── graph/            # Knowledge graph extraction, merging, scoring
│   ├── evaluation/       # Probing harness, retention metrics, visualizations
│   └── utils/            # Shared utilities
├── notebooks/            # Exploration notebooks (Phase 1 primarily)
├── data/
│   ├── synthetic/        # Synthetic personal memory datasets
│   └── sessions/         # Simulated session transcripts
├── experiments/          # Experiment scripts (one per experiment)
├── tests/                # Unit and integration tests
├── spec.md
├── architecture.md
├── environment.yml       # Conda environment
└── README.md
```

Code starts in notebooks for Phase 1 exploration, migrates to `paramem/` package from Phase 2 onward. Experiments are standalone scripts that import from the package.

## Integration Points

- **HuggingFace Hub:** Model downloads, tokenizer loading
- **wandb:** Experiment logging, metric visualization
- **PEFT:** Adapter creation, loading, switching, merging
- **bitsandbytes:** 4-bit quantization for QLoRA
- **NetworkX:** Graph operations (centrality, merging)


### AD-8: RAG Baseline with FAISS (Phase 4)

RAG pipeline uses the same embedding model already installed (all-MiniLM-L6-v2) for chunk retrieval. FAISS-CPU for vector search — lightweight, no GPU needed at our scale (hundreds of chunks). Falls back to numpy cosine search if FAISS install fails on WSL2.

The RAG pipeline is evaluation infrastructure, not a competing product. It exists to diagnose where parametric memory wins or loses vs retrieval.

### AD-9: Curriculum-Aware Replay (Phase 4)

Before each training cycle, probe the adapter on replay pool items to measure per-fact recall. Facts with low scores get higher sampling probability. This directly addresses the Phase 3b finding that replay pool samples only ~8% of accumulated facts per cycle — curriculum sampling ensures the hardest facts get more exposure.

Trade-off: probing adds latency (~10-30s per cycle). Acceptable given 42-min fast-iteration target.

### AD-10: Key-Addressable Replay (Phase 4)

Adapter weights are the single source of truth for all personal knowledge. No external corpus of training samples is maintained.

During the compression phase, each session's knowledge graph is stored in the adapter alongside a unique retrieval key. During replay, the model is prompted with each known key to reconstruct the associated graph triples from its weights. All reconstructions are merged with the new session's graph, and the adapter is retrained on the complete merged result. This turns every incremental consolidation step into a mini batch-mode operation.

Key insight: reconstruction does not need to be perfect. Facts that matter get reinforced through natural conversational repetition. Reconstruction noise causes unimportant facts to drift and decay — this IS the forgetting curve, emerging from the mechanism rather than being engineered.

This replaces an earlier design (periodic full-retrain sweeps on stored QA pairs) which contradicted the core architectural invariant: knowledge lives in weights, not in files.

### AD-11: Procedural Adapter Targets MLP Layers (Phase 4, live in server deployment)

The procedural adapter targets both attention layers (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`). Episodic and semantic adapters target attention only.

**Rationale.** Attention-only tunes *routing* — which context to attend to at inference time. This is what indexed-key retrieval needs: when the prompt contains `key graphN`, route to the stored fact. Facts stored this way are retrievable but the model's *representation* of them is unchanged. MLP targeting tunes *representation* — the persistent transformation applied to each token's hidden state. The interpretability literature locates factual associations and stylistic patterns predominantly in MLP feed-forward layers. Preferences and habits are persistent behavioral shifts, not keyed lookups, so they need MLP imprinting to take.

**Implementation.** `paramem/server/config.py::ServerAdapterConfig` carries a `target_modules` field per adapter. `_make_adapter_config` honours it — no more hardcoded list. `ServerAdaptersConfig` defaults procedural to `["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]` (attention + MLP). Overridable in `server.yaml`.

**Cost.** Procedural-only: ~3× more trainable params (~8 M → ~25 M at rank 8), ~30 MB → ~95 MB adapter file on disk, ~300–600 MB extra VRAM during training. Fits within the 8 GB budget alongside Mistral 7B NF4 + STT/TTS. Episodic and semantic unchanged.

Extraction uses a dedicated `extraction_procedural.txt` prompt for preference/behavioral content, separate from the factual extraction prompt.

### AD-12: Swappable Extraction Backend (Phase 4) — superseded

Originally posed as a `backend` parameter on `extract_graph()` to switch between a local 3B prompt-and-parse path and an API-based extractor. Superseded by AD-16: instead of swapping backends, extraction became a staged chain where the local model and the SOTA cloud model run *both*, in different roles. The local model owns transcript-touching stages (extract, anonymize); SOTA runs only on anonymized data (enrichment, plausibility). The `backend` parameter was never shipped.

### AD-13: Indexed Key Memory (Phase 4)

Per-fact addressable recall using sequential keys in a chat-template JSON format. Each fact is assigned a key (graph1, graph2, ...) and the model is trained to reconstruct that fact when prompted with the key. Training stays in the proven chat-template shape that avoids the format collision that destroyed F4.7 (raw structured triples in a different schema) — the framing is identical to Phase 1.

Key insight: numeric keys are more precise retrieval cues than natural language questions (90% vs 47% recall). The model learns the pattern `key → JSON` reliably at rank 8.

**Two encodings (migration in progress, gated by `consolidation.indexed_format`):**
- **QA-pair encoding** (`"qa"`, current default, being retired): the LLM QA generator (`paramem/graph/qa_generator.py`) mints a `(question, answer)` pair per graph triple; the adapter is trained on `key → JSON{key, question, answer}` plus a standalone natural-question example (2 training examples per fact). Recall template: `"Recall the QA pair stored under key 'graphN'."`
- **Quadruple encoding** (`"quad"`, going-forward): the adapter is trained directly on the merged-graph triple, `key → JSON{key, subject, predicate, object}` (1 training example per fact, no QA-generator LLM step). Recall template: `"Recall the fact stored under key 'graphN'."` The quad units come from `assign_quad_keys` over `merged_graph.relations + relation_prep._flatten_entity_attributes(merged_graph.entities)` (relations plus entity-attribute projections such as `has_email` / `has_phone`), partitioned to episodic / procedural by `relation_prep.partition_relations`, then formatted by `format_quadruple_training` (`paramem/training/quadruple_memory.py`). Round-trip-clean; ~½ the per-fact training cost.

A QA-trained adapter probed with the quad template fails (and vice versa), so the inference path reads each adapter's format and uses the matching template + parser. The default flips to `"quad"` after a production consolidation cycle proves it (the QA path stays available for the historical indexed-key tests).

### AD-14: SimHash Registry for Hallucination Detection (Phase 4)

An external SimHash registry (key → 64-bit fingerprint) is saved alongside each adapter. SimHash is a locality-sensitive hash (Charikar, 2002): similar content produces similar fingerprints, enabling continuous confidence scoring (0.0–1.0) rather than binary pass/fail.

Two-layer defense:
1. **Registry membership** (hard gate): keys not in the registry are untrained → reject immediately.
2. **Content fingerprint** (soft gate): compute SimHash of recalled content, compare to registry fingerprint via normalized Hamming distance. Confidence ≥0.75 → accept; below → reject.

Design constraints satisfied:
- Only 8 bytes stored per key (64-bit integer) — not training content.
- No modification to the JSON training format (3-field `{key, question, answer}` under the QA encoding, 4-field `{key, subject, predicate, object}` under the quadruple encoding); the fingerprint hashes the rendered content string either way (`f"{key} {question} {answer}"` or `f"{key} {subject} {predicate} {object}"`). Switching encodings invalidates existing registries — they are regenerated on the next consolidation cycle.
- Tolerates minor recall variations (e.g., casing differences score >0.8).
- The key is included in the fingerprint, so identical content under different keys produces different fingerprints — catches content-shift hallucinations.

Failed alternative: training a check hash into the JSON response caused format collision (0/10 recall). External registry is the only viable approach.

### AD-15: Indexed Key Consolidation Loop (Phase 4)

The consolidation loop integrates indexed key memory (AD-13) with the existing graph extraction and promotion pipeline. Each cycle: extract relations from session → assign sequential keys to new facts → train episodic adapter on all active keys → promote entities that meet recurrence threshold → train semantic adapter on promoted keys.

Key design decisions:
- **Capacity cap with eviction:** `max_active_keys` (default 20) limits adapter load. When exceeded, oldest keys are evicted. Keys survive eviction through reconstruction — if the adapter can still recall them, they're re-registered.
- **Dual-field promotion matching:** When checking if a keyed fact belongs to a promoted entity, both the subject and the object are checked (`source_subject` / `source_object` under the QA encoding, `subject` / `object` under the quadruple encoding). Reverse QA templates (e.g., "Who lives in Heilbronn?") swap subject/object, so both must be checked.
- **Periodic reconstruction:** Fidelity probing runs every N cycles (default 5), not every cycle. Per-cycle probing consumed 73% of cycle time in entity-replay experiments.
- **SimHash registry per adapter:** Each adapter (episodic, semantic) maintains its own SimHash registry. Keys promoted from episodic to semantic are registered in the semantic registry and removed from episodic.

Validated: 10-cycle smoke test, episodic 6/6 (100%), semantic 6/6 (100%), 49.9 min total.

### AD-16: Multi-Stage Privacy-Aware Extraction Pipeline (Phase 5)

Graph extraction is a staged chain built around a cloud-boundary privacy envelope.
The local model owns everything that touches real user data; the SOTA cloud model
sees only anonymized placeholders. Every stage falls forward — a failure at stage
N keeps the predecessor's output and continues.

1. **Extract** (`configs/prompts/extraction.txt`): local model emits triples + entities. Speaker-name injection pins the real speaker as canonical subject.
2. **Anonymize** (`configs/prompts/anonymization.txt`): local model produces `{real → placeholder}` mapping + anonymized transcript.
3. **Leak guard + repair** (`verify_anonymization_completeness`, `_check_mapping_totality`, `_repair_anonymization_leaks`): PII-scoped check with optional spaCy NER cross-check. Placeholders follow an **open-vocabulary shape contract** (`^[A-Z][A-Za-z]*_\d+$`); the prefix is derived from the upstream NER/extractor type via PascalCase rather than picked from a fixed lexicon. Missed names extend the mapping, hallucinated names drop referencing triples, and orphan placeholders are dropped at the residual sweep — the prior position-based pairing (`anon_facts[i] ↔ original_relations[i]`) was retired in favour of the totality diagnostic.
4. **SOTA enrichment with delta protocol** (`configs/prompts/sota_enrichment.txt`, `_filter_with_sota`): SOTA returns a delta envelope `{add, modify, drop, bindings}` — only the changes against the input fact list, plus `bindings: {placeholder: real_name}` for any net-new entities it introduced. The pipeline applies the delta to the input facts, merges bindings into the reverse mapping before de-anonymization, and reconstructs the updated transcript locally from the bindings. No transcript token-diff and no fact-echo: this replaced the prior "echo every fact + full transcript" envelope (≈ order-of-magnitude token reduction on small inputs).
5. **De-anonymize via state-machine substitution** (`_apply_bindings`): deterministic substring replacement of placeholder tokens with their real values from the merged reverse mapping. Zero local-LLM cost. Any fact whose subject or object still contains a placeholder-shaped token after substitution is dropped (residual-fact filter preserved, just no longer fronted by an LLM call).
6. **Plausibility** (`configs/prompts/sota_plausibility.txt` or `_local_plausibility_filter`): residual safety net. Grounding-based — every rule asks the judge to ground each fact against the transcript before deciding. Six rules cover (R1) self-loops, (R2) name-swap and role-leak shapes, (R3) transcript contradiction, (R4) conversation-role leaks (`Assistant`/`User`/...), (R5) content-free objects (`Unknown`/`None`/...), (R6) namespaced system identifiers (`media_player.X`). R3–R6 are described as categories rather than fixed token-lists, so the judge can extend them when the transcript supports the call. The closed-vocabulary R4 of the prior version was removed when the placeholder vocabulary opened up.

A **fallback path** (`_fallback_plausibility_on_raw`) runs local plausibility on the raw extraction when the primary chain empties out. Per-stage diagnostics (`SessionGraph.diagnostics`) record raw outputs, transcript round-trip, and dropped facts for audit.

**Single chokepoint.** Every orchestrator (training consolidation, calibration endpoint, experiments, tests) reaches the extraction chain through `ExtractionPipeline` (`paramem/graph/extraction_pipeline.py`) — one class that owns kwarg assembly, prompt-filename resolution, adapter guard, and gradient-checkpointing discipline. Direct calls to `extract_graph(...)` or `extract_procedural_graph(...)` are forbidden by `tests/test_extraction_pipeline_guard.py`. The class exposes `run(transcript, session_id, *, source_type, **overrides)` for transcript-shaped inputs and `run_procedural(...)` for the preference/habits stream; `kwargs(*, source_type, **overrides)` returns the resolved kwarg dict for callers that need to invoke `extract_graph` indirectly (e.g. legacy fixtures in the grandfathered list).

### AD-17: Background Training with Inference Pause (Phase 5)

Consolidation supports two code paths:

- **Blocking:** `run_consolidation` — extracts all sessions then trains under a single GPU lock. Used for manual `POST /consolidate`.
- **Scheduled (cooperative):** `_extract_and_start_training` spawns a `BackgroundTrainer` that releases the GPU lock per step so voice turns interleave. Driven by a systemd user timer whose period is derived from `consolidation.refresh_cadence × consolidation.max_interim_count`. `refresh_cadence` is the only user-facing knob — accepts `"HH:MM"` (daily), `"every Nh"`/`"every Nm"` (interval), `"daily"`, or `""`/`"off"` (manual only). `GracefulShutdownCallback` stops training at epoch boundaries on shutdown; a failed interim cycle is logged and the pending sessions are left for retry on the next tick. Optionally, `RecallEarlyStopCallback` (gated by `consolidation.recall_early_stopping`, default OFF) runs co-resident with `GracefulShutdownCallback` at the same `on_epoch_end` hook and fires `should_training_stop` once the staged adapter has memorized its full per-tier key set. The callback is constructed inside `ConsolidationLoop._maybe_make_recall_callback` and attached at every production-reachable `train_adapter` call site (4 in `paramem/training/consolidation.py`, 1 in `paramem/server/active_store_migration.py`).

Batch consolidation processes sessions as: `extract_session()` for all pending sessions, then `train_adapters()` once.

A **simulation mode** (`consolidation.mode: simulate`) persists the knowledge graph to disk as `graph.json` under `<adapter_dir>/<tier>/` instead of training LoRA weights. Both modes write to the same unified layout; the distinction is whether timestamped weight slot subdirectories are present alongside the graph. `DiskMemorySource` reads `graph.json` for simulate-mode recall; `WeightMemorySource` probes adapter weights for train-mode recall. Switching `consolidation.mode` between `train` and `simulate` triggers a per-tier active-store migration on next startup, gated by 100% recall — the source store is kept until the target is verified, so an interrupted migration falls back cleanly to the former mode (see `paramem/server/active_store_migration.py`).

### AD-18: Multi-Engine Multilingual TTS (Phase 5, F5.7)

Local text-to-speech via pluggable engines (`ENGINE_REGISTRY`) behind a common `TTSEngine` ABC:

- **Piper** (ONNX runtime): fast, high-quality voices for well-supported languages (en, de, fr, es). Sub-second synthesis on CPU.
- **MMS-TTS** (HuggingFace VitsModel): broader language coverage (e.g. Tagalog) where Piper has no voice model.
- **Kokoro-82M** (optional, opt-in per voice): higher-quality neural voices for en/fr/es and others (no German). Apache-2.0, CPU-capable.

`TTSManager` routes synthesis requests by language code to the configured engine/voice from `server.yaml` (per-voice device, CPU default). Exposed as a Wyoming protocol server (port 10301): it advertises `supports_synthesize_streaming` and handles `SynthesizeStart`/`Chunk`/`Stop`, which is what lets HA's streaming voice pipeline deliver the audio to satellites/Sonos (engines that don't advertise streaming get a degraded delivery path).

Language detection flows from two sources, both feeding the same resolver in `/chat`:

- **Voice path:** Whisper STT → `TranscriptionResult.language` → `_state["latest_language_detection"]` → `/chat` handler.
- **Text path:** fastText `lid.176` (`paramem/server/lang_id.py`) eager-loaded at server lifespan startup when `text_lang_detection.enabled`. Invoked on the request text only when no STT-derived signal is present and the request carries no voice embedding. CPU-only, zero VRAM cost; the 126 MB model is fetched once via `scripts/setup/download-langid-model.sh` into `~/.cache/paramem/lang_id/`. Disabled by default in the example config so deployments without the model file do not warn.

`_language_instruction()` injects "Respond in {language}" into system prompts for non-English input. Speaker profiles persist `preferred_language` for cross-session consistency on the voice path; the text path has no speaker-preference fallback (text `/chat` cannot identify a speaker without a voice embedding), so the detector's confidence threshold is conservative.

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

## Known Constraints and Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| 8GB VRAM limits batch size and sequence length | Slower training, potential quality impact | QLoRA + gradient checkpointing + gradient accumulation; monitor for quality issues |
| WSL2 CUDA memory reporting can be inaccurate | Unexpected OOM during training | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; keep training data on Linux filesystem |
| Multi-adapter simultaneous training not natively batched in PEFT | Must train adapters sequentially per consolidation cycle | Acceptable for PoC; each adapter trains independently anyway |
| Graph extractor quality depends on base model capability | Poor extraction → poor consolidation signal | Evaluate extraction quality early in Phase 3; consider separate extractor model if needed |
| Catastrophic forgetting in the episodic adapter during consolidation | Core technical risk — the thing we're trying to solve | Phase 2 is entirely dedicated to understanding and mitigating this |
| Promotion threshold tuning is empirical | Wrong thresholds → too much or too little promotion | Phase 3 includes sensitivity analysis; all thresholds configurable |
| RAG outperforms parametric on all metrics | Undermines project thesis | Expected for factual recall; parametric value is in compression and style — measure both |
| Key reconstruction quality degrades with many keys | Adapter capacity limits reliable reconstruction | Cap active keys at 50; retire empty keys; consolidate session keys into topic keys on promotion |
| Curriculum probing adds latency | Slows iteration cycle | Cap probe to 50 items, cache results, skip on smoke test runs |

## WSL2-Specific Configuration

- CUDA toolkit installed via conda (`nvidia::cuda-toolkit`), not system packages
- All data and code on Linux filesystem (`/home/...`), never `/mnt/c/...`
- Windows NVIDIA driver ≥560 required for WSL2 CUDA support
- Set environment variable: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
