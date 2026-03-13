# NeuroMem — Architecture Decisions (Phases 1–4)

## Chosen Tech Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | ML ecosystem standard, PEFT/HF native |
| **Environment** | Conda | User preference; manages CUDA toolkit cleanly on WSL2 |
| **Base Models** | Qwen 2.5 3B (primary), Llama 3.2 3B (secondary) | Best benchmarks at 3B (Qwen), largest community (Llama). Model-agnostic design allows swapping. |
| **Fine-tuning** | QLoRA via PEFT + bitsandbytes (4-bit) | Required for 8GB VRAM constraint |
| **Framework** | PyTorch + HuggingFace Transformers + PEFT + Accelerate | Industry standard, best LoRA multi-adapter support |
| **Graph Extractor** | LLM-based structured output (Phase 3) | Highest accuracy for relation extraction; use `outlines` or `instructor` for constrained JSON generation |
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
  ├── adapter: "episodic"  (rank 8, lr 1e-4)
  ├── adapter: "semantic"  (rank 24, lr 1e-5)
  └── adapter: "procedural" (rank 12, lr 5e-5)  [future]
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
- **outlines / instructor:** Constrained LLM generation for graph extraction (Phase 3)

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

### AD-11: Procedural Adapter Targets MLP Layers (Phase 4)

The procedural adapter (rank 12) targets both attention layers (q/k/v/o_proj) and MLP layers (gate/up/down_proj). This differs from episodic/semantic adapters which target attention only. Rationale: behavioral patterns (style, formatting) are more closely tied to the model's feed-forward computations than to attention patterns. Already configured in default.yaml.

### AD-12: Swappable Extraction Backend (Phase 4)

`extract_graph()` accepts a `backend` parameter to select between local (current 3B prompt-and-parse) and API-based extraction. The local backend has 94.5% success rate — workable but noisy. API backend is opt-in for users who want higher quality and have API access.

This is a strategy pattern, not a rewrite. The extraction prompt, schema, and normalization stay the same regardless of backend.

### AD-13: Indexed Key Memory (Phase 4)

Per-fact addressable recall using sequential keys in the proven QA chat-template format. Each QA pair is assigned a key (graph1, graph2, ...) and the model is trained on both indexed recall prompts and individual QA pairs. This avoids the format collision that destroyed F4.7 (structured triples) — the training format is identical to Phase 1.

Key insight: numeric keys are more precise retrieval cues than natural language questions (90% vs 47% recall). The model learns the pattern `key → JSON{key, question, answer}` reliably at rank 8.

### AD-14: SimHash Registry for Hallucination Detection (Phase 4)

An external SimHash registry (key → 64-bit fingerprint) is saved alongside each adapter. SimHash is a locality-sensitive hash (Charikar, 2002): similar content produces similar fingerprints, enabling continuous confidence scoring (0.0–1.0) rather than binary pass/fail.

Two-layer defense:
1. **Registry membership** (hard gate): keys not in the registry are untrained → reject immediately.
2. **Content fingerprint** (soft gate): compute SimHash of recalled content, compare to registry fingerprint via normalized Hamming distance. Confidence ≥0.75 → accept; below → reject.

Design constraints satisfied:
- Only 8 bytes stored per key (64-bit integer) — not training content.
- No modification to the 3-field JSON training format.
- Tolerates minor recall variations (e.g., casing differences score >0.8).
- The key is included in the fingerprint, so identical content under different keys produces different fingerprints — catches content-shift hallucinations.

Failed alternative: training a check hash into the JSON response caused format collision (0/10 recall). External registry is the only viable approach.

### AD-15: Indexed Key Consolidation Loop (Phase 4)

The consolidation loop integrates indexed key memory (AD-13) with the existing graph extraction and promotion pipeline. Each cycle: extract relations from session → assign sequential keys to new QA pairs → train episodic adapter on all active keys → promote entities that meet recurrence threshold → train semantic adapter on promoted keys.

Key design decisions:
- **Capacity cap with eviction:** `max_active_keys` (default 20) limits adapter load. When exceeded, oldest keys are evicted. Keys survive eviction through reconstruction — if the adapter can still recall them, they're re-registered.
- **Dual-field promotion matching:** When checking if a QA pair belongs to a promoted entity, both `source_subject` and `source_object` metadata are checked. Reverse QA templates (e.g., "Who lives in Heilbronn?") swap subject/object.
- **Periodic reconstruction:** Fidelity probing runs every N cycles (default 5), not every cycle. Per-cycle probing consumed 73% of cycle time in entity-replay experiments.
- **SimHash registry per adapter:** Each adapter (episodic, semantic) maintains its own SimHash registry. Keys promoted from episodic to semantic are registered in the semantic registry and removed from episodic.

Validated: 10-cycle smoke test, episodic 6/6 (100%), semantic 6/6 (100%), 49.9 min total.

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
