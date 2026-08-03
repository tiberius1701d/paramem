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
| spaCy + custom entity/relation extraction | Fast, deterministic, CPU-only | Requires training data, misses implicit relations | Fallback if LLM extraction too slow |
| GLiNER | Zero-shot entity extraction, lightweight | Entity extraction only, no relations | Potential component within a hybrid pipeline |

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

### AD-11: Procedural Adapter Targets MLP Layers (live in server deployment)

The procedural adapter targets both attention layers (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`). Episodic and semantic adapters target attention only.

**Rationale.** Attention-only tunes *routing* — which context to attend to at inference time. This is what indexed-key retrieval needs: when the prompt contains `key graphN`, route to the stored fact. Facts stored this way are retrievable but the model's *representation* of them is unchanged. MLP targeting tunes *representation* — the persistent transformation applied to each token's hidden state. The interpretability literature locates factual associations and stylistic patterns predominantly in MLP feed-forward layers. Preferences and habits are persistent behavioral shifts, not keyed lookups, so they need MLP imprinting to take.

**Implementation.** `paramem/server/config.py::ServerAdapterConfig` carries a `target_modules` field per adapter. `_make_adapter_config` honours it — no more hardcoded list. `ServerAdaptersConfig` defaults procedural to `["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]` (attention + MLP). Overridable in `server.yaml`.

**Cost.** Procedural-only: ~3× more trainable params (~8 M → ~25 M at rank 8), ~30 MB → ~95 MB adapter file on disk, ~300–600 MB extra VRAM during training. Fits within the 8 GB budget alongside Mistral 7B NF4 + STT/TTS. Episodic and semantic unchanged.

Extraction uses a dedicated `extraction_procedural.txt` prompt for preference/behavioral content, separate from the factual extraction prompt.

### AD-13: Indexed Key Memory

Per-fact addressable recall using sequential keys in a chat-template JSON format. Each fact is assigned a sequential key (`graphN` / `procN`) and the model is trained to reconstruct that fact when prompted with the key. Training stays in the proven chat-template shape that avoids the format collision produced by mixing QA pairs and hashes in a single adapter pass. Shipped default: real minting starts at `graph201` / `proc201` (`ConsolidationLoop._indexed_next_index` / `_procedural_next_index`), not `graph1` — the low band (`graph1`-`graph200`, `proc1`-`proc200`) is reserved unconditionally for the synthetic donor's training population (`paramem.training.donor`), which seeds every measured-cold fold as the unconditional standard mechanism (no config flag; see `benchmarking.md`, "Test 20").

**Key insight:** keyed retrieval is the reliable interface for parametric recall; un-keyed natural-language questions yield inconsistent results (see `benchmarking.md`, Test 5 / keyed vs. natural comparison). The model learns the pattern `key → JSON` reliably at rank 8.

**Two encodings:**
- **QA-pair encoding** (`"qa"`, legacy/test-only): the LLM QA generator mints a `(question, answer)` pair per graph triple; the adapter is trained on `key → JSON{key, question, answer}`. Recall template: `"Recall the QA pair stored under key 'graphN'."`
- **Quadruple encoding** (`"quad"`, production): the adapter is trained directly on the merged-graph triple, `key → JSON{key, subject, predicate, object}` (1 training example per fact, no QA-generator LLM step). Recall template: `"Recall the fact stored under key 'graphN'."` The quad units come from `assign_keys` over `merged_graph.relations + relation_prep._flatten_entity_attributes(merged_graph.entities)`, partitioned to episodic / procedural by `relation_prep.partition_relations`, then formatted by `format_entry_training` (`paramem/memory/entry.py`). Round-trip-clean; ~½ the per-fact training cost.

A QA-trained adapter probed with the quad template fails (and vice versa), so the inference path reads each adapter's format and uses the matching template + parser.

### AD-14: SimHash Registry for Hallucination Detection

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

### AD-16: Multi-Stage Privacy-Aware Extraction Pipeline

Graph extraction is a staged chain built around a cloud-boundary privacy envelope.
The local model owns everything that touches real user data; the cloud model
sees only anonymized placeholders. Every stage falls forward — a failure at stage
N keeps the predecessor's output and continues.

1. **Extract** (`configs/prompts/extraction.txt`): local model emits triples + entities. The session speaker's stable `speaker{N}` system id is injected as the canonical subject of their facts; the display name is passed as comprehension context only, and a name is substituted for the id only at the reply boundary, when a response is about to be shown or spoken to the user.
2. **Anonymize** — `configs/prompts/anonymization.txt` for the transcript-bearing session tier and chat egress, `configs/prompts/anonymization_facts.txt` (mapping only, no transcript machinery) for the transcript-free graph tier — via `paramem.cloud.anonymize.anonymize`, the ONE anonymize chain every cloud-bound path composes through: the local model is the sole classifier against the operator-configured `sanitization.scrub` PII-vocabulary hints — no code-side entity-type gate — and returns TWO artifacts: the `{real → placeholder}` mapping and its own rewrite of the transcript with in-scope values placeholdered (`anonymized_transcript`). The chain still builds the anonymized fact array deterministically (one entry per `graph.relations`, `subject`/`object` substituted through the mapping via an edge-aware, case-sensitive `_substitute_whole_words`, `predicate`/`relation_type`/`confidence` copied verbatim — the predicate is never a substitution target), so a fact can never be lost, reworded, or dropped by the anonymizer, and a placeholder can never be glued into a predicate at this stage. The transcript, by contrast, is authored by the model rather than mechanically rebuilt from the mapping: prose classification is context-dependent (a name and a common-word homograph share identical bytes — "Will" vs "Will I?") in a way the case-sensitive fact fields are not, so only the model, holding that context, can rewrite it correctly. A parse failure; a missing/empty `anonymized_transcript` returned over a non-empty input transcript when the model's own mapping named something (the one inconsistent shape that fails closed — an empty/missing rewrite is otherwise legitimate: nothing to rewrite when the input transcript is empty, or when the model's own mapping came back empty, in which case the chain proceeds on the ORIGINAL argument transcript, the same "nothing in scope" verdict already accepted on the facts surface); or (for the graph tier — see below) a call where the model named real content but nothing survived to the final table — fails closed (`AnonymizedContract.status == "failed"`) — callers never fall back to the original real-name transcript on an actual fail-closed verdict. Placeholders follow an **open-vocabulary shape contract** (`^[A-Z][A-Za-z]*_\d+$`) — except the speaker's own `speaker{N}` anchor, which is deliberately exempt (already anonymous, never minted, never re-braced) and can appear as a forward-map value without matching that shape.
3. **Entity-surface correction** (`configs/prompts/entity_correction.txt`): the local model reviews real entity surfaces on the anonymization reverse map and node attributes and corrects misspelled place/org/concept names; an apply-gate rejects any proposal targeting an entity not already known, and all verdicts — accepted and rejected — are recorded on `graph.diagnostics["entity_correction_verdicts"]` (persisted as a debug artifact when debug is on). Speaker/person nodes are left untouched.
4. **Cloud enrichment with delta protocol** (`configs/prompts/cloud_enrichment.txt`): cloud returns a delta envelope `{add, modify, drop, bindings}` — only the changes against the input fact list, plus `bindings: {placeholder: real_name}` for net-new entities. The pipeline applies the delta, merges bindings before de-anonymization, and reconstructs the updated transcript locally. No transcript token-diff and no fact-echo (order-of-magnitude token reduction vs. the prior "echo every fact" envelope). `add`/`modify` entries are restricted to the fields that actually reach a relation (subject/predicate/object/relation_type/confidence/symmetric) — any other key an LLM invents is stripped before the entry enters the pipeline, so it can never later be mistaken for an unresolved placeholder. Rejection is per-action, never whole-delta (2026-07-22 cloud-admission redesign, `_apply_enrichment_delta`): an `add` naming a token neither in the anonymized facts/transcript cloud was shown nor in its own `bindings` (orphan) is dropped; a `modify` whose `fields` would introduce one is discarded and the pre-enrichment fact is kept unchanged instead; `drop` is honored unconditionally. A binding whose key collides with the local map is purely informational (`graph.diagnostics["cloud_binding_collisions"]`) and never a rejection reason — the local value always wins on resolution. Per-cycle counts and the distinct rejected tokens are operator-visible in `graph.diagnostics["cloud_enrichment_report"]` plus a WARNING-level log line.
5. **De-anonymize** (`_apply_bindings`): the single deanon exit gate, in three ordered steps. First, a **predicate invariant** run BEFORE substitution drops (never repairs) any fact whose `predicate` field contains a token from the declared placeholder vocabulary (the union of the anonymizer's `reverse` map and cloud's `bindings`) — the predicate is never a substitution target, so checking it after substitution would silently miss an already-corrupted predicate. Second, **substitution**: deterministic substring replacement of placeholder tokens with their real values, resolved against an observed-scoped map — the local (real) mapping for tokens cloud was actually shown, cloud's own `bindings` for tokens it minted, with the local mapping always taking precedence on conflict. Third, a **residual sweep** checks every fact field (subject/predicate/object/relation_type/confidence/symmetric — never a non-fact field an LLM invented) against the same declared vocabulary, plus a placeholder-shaped-token regex as a last-resort, fail-closed backstop for an undeclared orphan the vocabulary check can't see; the regex is never load-bearing for resolution or substitution, only for this final net. Both steps are fail-closed (drop, not repair) and both record counters/lists in `graph.diagnostics` (`predicate_placeholder_dropped` / `predicate_placeholder_dropped_facts` for the predicate invariant, written at the deanon stage — a placeholder glued into a predicate can only arrive in cloud's *returned* facts now, since the anonymizer stage never produces facts at all — and `residual_dropped_facts` for the residual sweep) — disjoint categories, never double-counted.
6. **Plausibility** (`configs/prompts/cloud_plausibility.txt` or `local_plausibility_filter`): grounding-based residual safety net. Six rules cover (R1) self-loops, (R2) name-swap and role-leak shapes, (R3) transcript contradiction, (R4) conversation-role leaks, (R5) content-free objects, (R6) namespaced system identifiers.

A **fallback path** runs local plausibility on the raw extraction when the primary chain empties out. Per-stage diagnostics record raw outputs, transcript round-trip, and dropped facts for audit.

**No post-hoc leak check between steps 2 and 4** (removed — see SECURITY.md for the residual this leaves and why it was deleted, not weakened). The anonymized FACTS cloud sees are built entirely by the script in step 2 from a table it mints and owns (`_build_anonymization_mapping`), substituted through `_substitute_whole_words` — an exact, case-sensitive primitive; there is nothing left for a post-hoc scan to verify on that surface. The anonymized TRANSCRIPT cloud sees, by contrast, is authored by the model itself in step 2 — a post-hoc scan there would just be re-verifying the model's own classification judgment, which this design deliberately does not do at runtime (see SECURITY.md); that surface is instead verified offline, at the calibration gate. What still needs a runtime check is what cloud sends *back* (step 5), which is a model rewriting content, not a table the script owns.

**Second call site — graph-tier enrichment.** The privacy envelope above (steps 2–6) is not session-tier-only. `paramem.training.graph_enrich.enrich_graph`'s post-merge, cross-session cloud pass (`paramem.graph.extractor.request_graph_enrichment`) runs the SAME anonymize → cloud → de-anonymize chain, through `paramem.cloud` (`anonymize.py` / `deanonymize.py`) — the one round-trip contract every cloud-egress path (session-tier extraction, graph-tier enrichment, chat egress, and their calibration harnesses) composes through (`paramem.cloud.placeholders` remains the model-free, IO-free primitive kit `paramem.cloud` is built from), before any subgraph triple leaves the process.

The cumulative fold graph carries no reliable entity types of its own (registry-derived relations have none), and this pass does not derive one: before each chunk's cloud call it runs `anonymize` (the SAME chain session-tier extraction uses) over the chunk's triples, passing `identity_domain=chunk_nodes`. `anonymize` reconciles the local anonymizer's mapping keys onto the chunk's actual node-key text via `canonical()` internally — a re-cased/separator-varied/diacritic-varied key from the local model (e.g. `"Yang Ming"`) is re-keyed onto the node it names (e.g. `"yang ming"`) with the model's own placeholder preserved verbatim, and an entry matching no node in the chunk (or an ambiguous multiple) is dropped and counted (`AnonymizedContract.rekey_dropped`, surfaced as `mapping_rekey_dropped`). This reconciliation is identity reconciliation, not classification — not inside the shared `_substitute_whole_words` primitive, which matches exactly everywhere, including at this tier. `request_graph_enrichment` receives the already-built `AnonymizedContract` and applies no scope gate of its own on the OUTBOUND side. The RESPONSE side has no whole-chunk gate either (2026-07-22 cloud-admission redesign, retiring the prior `totality_rejected_chunks` behaviour): every `relations` entry here is effectively an `add` (this tier has no local baseline to preserve), so `deanonymize_facts`'s fail-closed residual sweep in `_apply_bindings` simply drops the individual relation(s) it cannot resolve post-substitution, counted in `dropped_relations`.

A local mapping that comes back completely empty is a legitimate "nothing in scope" verdict and proceeds. A local mapping that DID name real (non-speaker) content but nothing survived to the final table — whether dropped by the model's own placeholder-shape validation or by the node-key reconciliation above — is a classification/identity-match failure: the affected facts are withheld from that chunk's outbound cloud call rather than sent unmasked, while any of the chunk's other facts that classified successfully still reach it. Only when nothing in a chunk survives classification does `anonymize` fail closed for the whole chunk (`status == "failed"`) and the chunk's cloud call itself is skipped, counted in `privacy_skipped_chunks`. Under the default `sanitization.scrub` (person name, email address, phone number, physical address, social profile URL), the anonymous `speaker{N}` handle is never tokenised at this tier either — the local anonymizer prompt forbids mapping it, so it reaches the payload bare by design (it carries no identifying information). **Operator opt-out**: an explicitly empty `sanitization.scrub` short-circuits before any model call — the chunk's triples egress to the cloud VERBATIM, the same opt-out contract every other cloud-egress path honours (see SECURITY.md for the privacy-posture note).

A local resource fault (insufficient free VRAM) during this pass degrades the PASS, not just the one chunk in progress: `enrich_graph` stops processing further chunks (`aborted_reason="vram"` in its returned diagnostics), keeps whatever chunks it already merged, and returns normally rather than raising — the chunk in progress when the fault occurred contributes nothing, but chunks that already completed keep their enrichment relations. `ConsolidationLoop._refine_consolidation_graph` records an `enrichment_degraded` incident and the fold proceeds to train on the merged-but-unenriched graph; enrichment self-heals at the next **full** fold — this pass is full-fold only (see AD-15 below), so recovery does not happen at an intervening interim cycle. The incident clears itself on that recovery: an enrichment pass that runs to completion resolves it, so the attention row on `/status` reflects the current state rather than the worst state ever reached. When the operator has cloud egress disabled entirely, a completed pass can never happen — so the incident instead clears the next time a session is processed, carrying a recorded reason that distinguishes "resolved by a clean run" from "resolved because cloud is off," rather than riding on `/status` forever with no path back to green.

The response's `relations` and `same_as` pairs are de-anonymized via `deanonymize_facts` / `deanonymize_text` — `paramem.cloud`'s exit gates, `observed`-scoped to the exact triples JSON sent to cloud — before `enrich_graph` ever consumes them — load-bearing for the speaker-pair guard (`is_speaker_id`), which cannot recognise a placeholder token as a speaker id. Accepted consequence: person-level `same_as` coreference (nickname/honorific variants of the same person) is lost under the default `scrub`, since both surfaces collapse to opaque tokens before the model sees them; org/place/thing coreference is unaffected (those surfaces stay verbatim under the default `scrub`).

**Single chokepoint.** Every orchestrator reaches the extraction chain through `ExtractionPipeline` (`paramem/graph/extraction_pipeline.py`). Direct calls to `extract_graph(...)` or `extract_procedural_graph(...)` are forbidden by `tests/test_extraction_pipeline_guard.py`. The class exposes `run(transcript, session_id, *, source_type, **overrides)` for transcript-shaped inputs and `run_procedural(...)` for the preference/habits stream.

### AD-15: Indexed Key Consolidation Loop

The consolidation loop integrates indexed key memory (AD-13) with the existing graph extraction and promotion pipeline. Each cycle: extract relations from session → assign sequential keys to new facts → train episodic adapter on all active keys → during the full consolidation fold (`ConsolidationLoop.consolidate`), keys whose per-key `reinforcement_count` meets the promotion threshold are promoted episodic→semantic; `store.move(key, "semantic")` moves the registry entry and SimHash — so promotion happens before tier assignment.

**Transcript-stage boundary (architectural symmetry).** The consolidation fold has two venues that run the SAME stage spine over the SAME input. The input is the in-RAM memory store (`MemoryStore`) in both: registry-true relations for every active key, across the main tiers and every interim slot. The venue is selected by `consolidation.mode` and is carried through the spine as a structural `FoldScope.source` (`weights` | `disk`) — never as a mode string. What the venue selects:
- **`train` (`source="weights"`)**: additionally probes the adapter weights to compute the recall-miss set, backs up the main tiers, retrains `episodic` / `semantic` / `procedural`, and persists + verifies the weights.
- **`simulate` (`source="disk"`)**: skips those weight-only blocks — there are no adapter weights — and persists each main tier as `<adapter_dir>/<tier>/graph.json`, the projection `DiskMemorySource` reads back when the store is next hydrated.

Everything else is one code path in both venues: materialize → refine (enrich / normalize) → promote → build keyed entries → drift partition → registry rewrite → persist → interim reap → router reload → tier delta. Both venues run `canonical()` node identity + Case-1/Case-2 dedup via `GraphMerger.merge(additive=True)` + `GraphTierRefiner.run_enrichment` (cross-session second-order relations + `same_as` coreference, cloud-cloud) then `GraphTierRefiner.run_normalization` (predicate-synonym collapse via `normalize_predicates`; runs when `refinement_normalization` is on, which is the default) — enrichment runs first so normalization collapses any cloud-coined predicate synonym before the fold's key assembly mints keys from the graph. Both passes are **full-fold only**: the interim scope's `FoldScope.enrich` / `FoldScope.normalize` are pinned `False` structurally, regardless of `refinement_enrichment` / `refinement_normalization` / `cloud_enabled` — see AD-10 below. Grooming logic is shared across scopes too: the interim tick (`run_consolidation_cycle`) and the full fold (`ConsolidationLoop.consolidate`, the single public fold entry) both route through the private spine `_run_fold`, and every persist tail — either scope, either venue — goes through the one `_persist_fold` dispatch. There is no dual-method parity requirement — a grooming change goes in `_run_fold` once and both venues inherit it. The fold has no notion of who asked for it: whether there is anything to consolidate at all is decided in the server's dispatch layer before the fold is entered. `POST /reconsolidate` is the on-demand re-grooming pass — it runs the same fold over a narrower key source (the main tiers' own keys), so it re-grooms and re-learns main memory without absorbing or reaping the interim slots.

**Persist and reap share one guard.** The interim slots a fold consumed are reaped only when that fold actually persisted its merged main tiers. A fold that rebuilt nothing writes nothing and reaps nothing, so the slots' content is never destroyed before a durable copy of the merge exists.

**Fold merge input is registry-true, in both venues.** The fold sources its Stage-2 merge input from `store.get(key)` / `store.bookkeeping_for_key(key)` (registry-true SPO) for every active key — never from the reconstruction result, and never from a direct disk read. Reconstruction exists only in the `train` venue and is a **health/retry signal**: a key whose reconstructed SPO disagrees with its registry-true SPO is flagged in `result["recall_miss_keys"]` and retrained with its registry-true content — it is never silently dropped. A recall miss does not delete a key. In the `simulate` venue there is no reconstruction and `recall_miss_keys` is always empty.

Key design decisions:
- **Capacity / passive decay:** `max_active_keys` (default 100000) imposes no practical limit; keys are not evicted by age. Unreinforced keys passively decay: those not re-seen for `decay_window` cycles are logged as decay candidates but are never actively deleted. Reconstruction noise causes unimportant facts to drift over time — this is the forgetting curve emerging from the mechanism. Validated to 550 keys with no observed ceiling.
- **Periodic reconstruction:** Fidelity probing runs every N cycles (default 5), not every cycle. Per-cycle probing consumed 73% of cycle time in entity-replay experiments.
- **SimHash registry per adapter:** Each adapter (episodic, semantic) maintains its own SimHash registry. Keys promoted from episodic to semantic are registered in the semantic registry and removed from episodic.

Validated: 10-cycle smoke test, episodic 6/6 (100%), semantic 6/6 (100%), 49.9 min total.

### AD-10: Key-Addressable Replay

Adapter weights are the single source of truth for all personal knowledge. No external corpus of training samples is maintained.

During the compression phase, each session's knowledge graph is stored in the adapter alongside a unique retrieval key. During the full consolidation fold, the model is prompted with each known key to reconstruct the associated graph triples from its weights. Reconstruction acts as a **health and retry signal**: a key whose reconstruction disagrees with its registry-true content is flagged for retrain but is never deleted by a miss. The fold's merge input is sourced from registry-true (subject, predicate, object) for every active key; reconstruction cannot manufacture a false dedup collapse. The adapter is retrained on the complete registry-true set.

**Dedup is registry-true.** Two keys collapse iff their registry-true SPO is identical. The collapsed key is **soft-staled** (registry entry retained, simhash retained, excluded from training) so the fact is still accessible to the stale-echo research seam and key ids can be recycled later. The surviving key inherits the standing of every key merged into it, so a fact that had earned its way into the semantic tier keeps that status through a collapse instead of being served from episodic again. The fold is **additive and lossless** with respect to registered facts: no registered fact is silently erased by a recall miss, and none is silently demoted by a merge.

Dedup also fires at the interim mini-fold, not only at the full fold: a session that recites a fact already stored in a main tier, or already keyed in an earlier interim slot still awaiting the next full fold, is deduped against the recalled, session-scoped facts from either source, so the recital never mints a transient interim key. The recital instead credits the surviving key's reinforcement count, exactly as a full-fold collapse would — provided it comes from a later session than the one that key was last seen in. Repetition within a single conversation is not reinforcement, so it does not raise the count. The interim fold merges these dedup targets — main-tier or sibling-interim — for Case-1 adoption and reinforcement credit only — they are excluded from the training set — and the interim fold runs no graph-tier refinement (enrichment or normalization) at all; both passes are full-fold only (see AD-15 above).

Key insight: reconstruction does not need to be perfect. Facts that matter get reinforced by coming up again in a later conversation — repetition inside one conversation does not count. Decay is passive: keys not re-seen for `decay_window` cycles are logged as decay candidates; there is no active deletion.

This replaces an earlier design (periodic full-retrain sweeps on stored QA pairs) which contradicted the core architectural invariant: knowledge lives in weights, not in files.

## Training Contract

**AD-7: Phased Code Structure** — exploration in notebooks; production code lives in the `paramem/` package (notebooks are exploration-only). Project structure is documented in `README.md`.

### AD-6: QLoRA Training with Gradient Checkpointing

8GB VRAM on the RTX 5070 requires:
- 4-bit quantization of the base model (bitsandbytes NF4)
- Gradient checkpointing enabled
- Batch size 1, gradient accumulation steps 8–16
- Sequence length capped at 512 tokens (safe), 1024 (stretch)
- `bfloat16` compute dtype (Blackwell architecture native)

These constraints are encoded as defaults in the training config, overridable per-experiment.

### AD-20: Staging+Promote Adapter Contract

Every adapter training event — consolidation cycle, interim mint, base-swap Phase B — runs through a two-slot **staging+promote** contract, not directly on the production tier. The contract has one entry point (`paramem/training/trainer.py::train_adapter`) and one staging slot per process (`in_training`).

**Two-slot rationale.** Mutating production weights in place is unsafe across two failure modes: (1) crash mid-training would leave the production slot in a half-trained state with no rollback path; (2) the recall sanity gate can reject the trained adapter (recall < 1.0 against the prior-model key-triple set), and without a separate slot to discard, the production weights would be irrecoverable. Production stays byte-identical to the last committed state until training completes, the recall gate passes at 1.0, and the new weights have been promoted by an explicit `copy_adapter_weights(staging → production)` step.

**Staging slot lifecycle.** The slot is transient — it exists only while a training event is in flight. Each training entry creates a fresh `in_training` slot (LoRA-init, seeded RNG when the target adapter is new); when the target adapter already exists, the slot instead starts from the production adapter's current weights via `copy_adapter_weights(production → in_training)` — this is what makes every scheduled fold warm by default. HF Trainer mutates the slot while production is untouched; on success the slot is promoted then deleted; on abort the slot is deleted and `staging_resume.json` + HF Trainer checkpoint are preserved for resume. The slot does not persist across training events.

**Consolidation vs. migration asymmetry.** Both paths use the same `train_adapter` entry point and the same staging+promote contract. They diverge in the starting weights:
- **Consolidation:** production weights at training entry are the previous cycle's promoted state; `copy_adapter_weights(production → in_training)` carries them into staging. Incremental — every cycle builds on the previous cycle's adapter.
- **Base-swap migration:** the production tier is explicitly reset to LoRA-zero before `train_adapter` is called. Training is from scratch on the new base model (LoRA weights of the old base do not transfer across different layer dimensions).

**Pause and resume.** "Pause" is process exit. On the next boot PEFT loads production from disk; `in_training` is absent (never persisted; excluded from backup). The next `train_adapter` call creates a fresh staging slot and `_resolve_resume_checkpoint` finds the saved checkpoint; HF Trainer's `resume_from_checkpoint` loads its weights into staging before continuing from step/epoch N+1.

**Live-reload after base-swap final tier.** After the final `migrate()` returns, the orchestrator calls `_live_reload_base_model` before marking `status=pass`. The reload tears down the PeftModel and rebuilds it from disk, picking up every tier's promoted adapter so the running server serves the new base without a systemctl restart. For the reload to fit on 8 GiB, all base-model holders (`BackgroundTrainer.model`, `ConsolidationLoop.model/.extraction.model`) are released via their encapsulated `release()` methods before the reload.

### AD-17: Background Training with Inference Pause

Every consolidation trigger — the systemd timer and all three operator endpoints — goes through one arbitrator, `_dispatch_consolidation`. It is **non-blocking in every case**: the arbitrator decides, submits the run to an executor, and returns immediately with a `status` and the `action` it resolved to (`interim`, `full` or `reconcile`). Progress is observed via `GET /status` (`consolidating`). Nothing runs the fold on the request thread.

The arbitrator owns three decisions the fold itself knows nothing about:

- **Who may run:** a busy server (fold in flight, chat in progress, GPU held, cloud-only) returns `deferred_*`; a migration TRIAL is checked both at the REST boundary and by the arbitrator itself (`_consolidation_dispatch_guards` → `deferred_trial_active`). Every REST door (including `POST /scheduled-tick`) returns 409 `trial_active` during a TRIAL. The boot-completion catch-up (below) dispatches in-process rather than through a REST call, so the arbitrator-level check is what makes it defer during a TRIAL instead of running.
- **What to run:** `AUTO` is requested by `POST /scheduled-tick` and by the boot-completion catch-up task described below — it becomes `FULL` or `INTERIM` via `_is_full_cycle_due`'s deadline math. `POST /consolidate` requests `FULL` directly, `POST /consolidate/interim` requests `INTERIM` directly, and `POST /reconsolidate` requests `RECONCILE` directly — none of them ever resolves `AUTO`, so none of them consults the deadline math or falls back between `FULL`/`INTERIM`.
- **Whether to run at all:** the **catch-up gate** (a scheduled tick that is not yet due against its own cadence mark) and the deadline resolution above are gated on `action is AUTO`, so both belong to the schedule alone — a directly requested `FULL`/`INTERIM` skips past them entirely: it means "now", not "if due". The **content gate** (nothing new to consume → `noop_*`, no GPU work) is a different property, checked per action: `FULL`'s content is any payload-bearing interim slot on disk (checked regardless of the CURRENT `max_interim_count`, so a slot minted before an operator lowered it to 0 is still absorbed and reaped rather than stranded) or, only at `max_interim_count == 0`, pending NAMED sessions; `INTERIM`'s content is pending NAMED sessions. It applies identically whether `FULL`/`INTERIM` was resolved from `AUTO` or requested directly — a manual door drops only the TIME condition, never the CONTENT condition. A `noop_*` status is information, not a refusal. `RECONCILE` is the one action exempt — the operator's explicit rebuild-the-store door — since its input (the main tiers' own stored keys) always exists; there is no bypass flag, the exemption is `RECONCILE` never reaching the gate (it still passes through the shared safety guards above). Session triage — retiring what can never be attributed — is a side-effect pre-stage that runs on *every* dispatch, so `RECONCILE` bypassing the content gate never switches orphan retirement off for that door. A directly requested `FULL`/`INTERIM` dispatch does not move the cadence window; the next scheduled tick still has its own content gate and noops on its own if the manual run consumed everything.

`FULL` and `RECONCILE` run the same fold and differ only in its **key source**: `FULL` folds every active key, interim slots included, and reaps the slots afterwards; `RECONCILE` rebuilds the main tiers from their own keys and leaves the interim slots and pending sessions exactly where they are. Interim disposal follows from the key source rather than from a flag of its own — a fold that did not absorb the slots must not reap them.

The same key-source distinction decides each main tier's **init policy**. `FULL` folds and interim folds train warm from the resident adapter's weights (see AD-20's staging-slot warm copy); `RECONCILE` deletes and re-initialises each tier before training — a cold rebuild, which is also the mechanism that eventually removes a registry-erased key's residual weight encoding (forgetting is registry-level: the SimHash serve-gate makes an erased key unservable immediately, and `RECONCILE` is the operator-invoked door that later trims its weight). A resident adapter whose LoRA config (rank, alpha, target modules) no longer matches the tier's configured LoRA topology is recreated cold regardless of the fold's key source. A third cold path is the recall-gate rejection itself: when an interim fold's post-save recall probe rejects the slot, the fold deletes it from VRAM as well as disk, so a same-window retry re-mints and re-enters cold rather than warm-starting from the rejected weights.

Below the arbitrator the training layer has **no notion of who asked**: `ConsolidationLoop.consolidate(mode=..., keys_from=...)` takes the fold's venue, its key source and its fold inputs, nothing else.

The **cooperative training path** (`_extract_and_start_training`) spawns a `BackgroundTrainer` that releases the GPU lock per step so voice turns interleave. Scheduling is driven by a systemd user timer whose schedule IS `consolidation.refresh_cadence` — the timer never sees the derived full period (`refresh_cadence × max_interim_count`, or `refresh_cadence` itself at `max_interim_count: 0`); that derivation is consumed by `_is_full_cycle_due`, which decides per-tick whether a fired timer runs a full fold or an interim cycle. `refresh_cadence` accepts `"HH:MM"` (daily), `"every Nh"`/`"every Nm"` (interval), `"daily"`, or `""`/`"off"` (manual only). Every rendered timer is `OnCalendar` + `Persistent=true`, so a tick missed during suspend/power-off fires again once systemd resumes — but a tick that fires while the server is still starting has nowhere to land yet (uvicorn hasn't bound the port) and would otherwise be lost. The server closes that gap itself: once its own startup finishes, it checks for a missed cycle and dispatches through the identical `AUTO` door `POST /scheduled-tick` uses. Both the systemd-fired tick and this boot-completion check are gated by the same durable last-attempt stamp (`paramem/server/schedule_state.py`), for every cadence kind — anchored (daily/weekly/HH:MM) and exact-divisor intervals included, not only intervals that don't divide evenly into a day/hour — so a duplicate or redundant tick inside the same mark's window is a no-op (see `paramem/server/systemd_timer.py` module docstring). The first-ever scheduled tick on a fresh deployment seeds the stamp and does not fold. The same boot-completion task runs a missed scheduled backup before any missed consolidation catch-up (so a backup never captures a fold's own output as though it predated the fold), then reconciles both the consolidation and scheduled-backup timers last, off the event loop — both timers reconcile from the same helper on every config apply that changes either, not only at server boot.

**`window_stamp` is provenance only.** Main adapter slots are stamped with the full-consolidation window they belong to, and that stamp is written into the slot manifest — but no code compares stamps to decide anything. `_is_full_cycle_due` never reads it. There is consequently no "clear the stamp to force a full cycle" escape hatch (it never worked); the way to run a full cycle on demand is `POST /consolidate`, and the way to rebuild main memory from its own stored knowledge is `POST /reconsolidate`.

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
     of `configs/prompts/pa_voice.txt`. The prompt is name-free: the
     identity-injection helpers (`_build_speaker_prefix`,
     `_build_system_prompt`) used by the local reasoning leg are not
     invoked for classification, so no speaker identity reaches the
     classifier system message. ~2-4 forward passes
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
the encoder + exemplar bank. When intent cannot be positively
established — below margin, or encoder/exemplars fail to load — the
query is classified `UNKNOWN`: no personal-memory access, and not
blocked from escalation, so it routes through the normal HA → cloud →
base-model chain. A classifier unavailable in a mode that requires it
raises an operator-visible incident, so the degraded state is loud,
not silent.

**State signal asymmetry.** PA graph match is intentionally NOT a
state signal here. Speaker enrollment must not classify the
speaker's own queries as `PERSONAL` (the old "speaker-in-graph →
PERSONAL" short-circuit caused imperatives from enrolled speakers
to misroute into the PA path). The router scopes keys by speaker
but lets the classifier decide intent.

### AD-21: One Cloud Master Switch, One Personal Verdict, One Egress Funnel

**One switch.** `cloud.enabled` (`CloudConfig`) is the single on-off for all
cloud egress: the conversation agent, the per-session extraction enrichment
chain, the graph-tier enrichment pass, and `/calibrate/enrich`. It replaced
two structurally disjoint switches that answered the same question with no
cross-reference between them (`consolidation.cloud_enabled` and
`agents.cloud.enabled`) — a deployment could have the conversation agent live
while the pipeline believed cloud was off, or the reverse. `agents.cloud` and
`agents.cloud_providers` carry provider, model and credentials only; they have
no on-off of their own. The switch is necessary but never sufficient: whether
a specific call may be placed is decided by
`paramem.cloud.admission.evaluate_cloud_egress`, which also requires a
supported provider, a model, a resolvable API key and (for OpenAI-compatible
providers) an endpoint. With no provider and no API key there is no cloud
mode; only the local model answers. Ship default is `false` — enabling it
sends knowledge-graph content to a third party under best-effort
anonymization only.

**Self-hosted is not cloud.** `admission.py`'s provider tables are the
registry of what "cloud" means. A host that speaks the OpenAI-compatible wire
format but runs on the operator's own hardware has no entry there and never
reaches an admission check.

**One personal verdict.** The intent classifier is the routing authority for
whether a turn is personal. A single self-reference check (encoder-based,
first-person fallback) supplements it for first-person queries that name
nothing the classifier keyed on. The two are unioned into one `is_personal`
verdict, computed once in `handle_chat` and threaded from there. An earlier
graph-anchored scrub — matching a query against the speaker's stored entity
surfaces — was removed: it re-derived a signal the classifier already owns and
could only add false positives. The sanitizer has no policy knob of its own:
what to DO about a personal verdict is the caller's decision. Self-referential
history turns are always dropped from a cloud payload, never
warned-and-passed.

**The verdict gates cloud, not HA.** HA is local and stays reachable as a
tool fallback on every path.

**The forwarded query is a distinct artifact.** The text after `[ESCALATE]`
is authored by the local model after it has recalled facts from parametric
memory, so it can carry personal content the user never typed. It gets its
own verdict from the same `is_self_referential` predicate, computed in
`_maybe_escalate`; a self-referential forwarded query suppresses the HA hop as
well as the cloud hop, because `ha_agent_id` is operator-pointed and may be
cloud-backed.

**One egress funnel.** `answer_via_cloud` is the sole cloud-egress entry
point, in both local and cloud-only mode; `cloud_mode`
(`block`/`anonymize`/`both`) applies through it. Forced routing
(`route=cloud:<provider>`) selects the provider, not a policy bypass. The
funnel branches on one question — can the local model anonymize? In local mode
it can, and the `cloud_mode` policy runs. In cloud-only mode there is no local
model: the memory store is absent, no ParaMem-held knowledge can reach the
cloud by any path, and the current turn egresses verbatim only if the operator
has permitted the cloud leg. Cloud-only is therefore honestly a plain cloud
agent — no intent classification, no personal-referent gate, no anonymization
— gated by the master switch and the degraded-serving decision, not by
`cloud_mode`.

**Degraded serving is an explicit operator decision.**
`cloud.allow_degraded_serving` (default `false`) gates the cloud leg when the
server is cloud-only for an *involuntary* reason — GPU held by another
process, insufficient VRAM, a failed adapter reload or apply, a persistent
CUDA fault. Deliberate cloud-only (`cloud_only: true`, `POST /gpu/release`)
and transient internal states (training, live reload) proceed regardless.
When the gate closes, the cloud leg closes and the HA leg stays open: HA
carries no ParaMem-held knowledge and runs on the user's own network, so
breaking it during a GPU conflict buys no privacy. Anything HA cannot serve
returns the canned limited-mode response. When the gate is open, the first
turn of each conversation on that path is prefixed with a notice that a cloud
model is answering — app-layer prefix, the same mechanism as the greeting,
never written to the session buffer and so never able to reach a training
transcript.

**The HA agent must be local.** `ha_agent_id` names the leg that stays open
when the cloud leg is closed, and ParaMem sends it cleartext. Pointing it at
a cloud-backed HA conversation agent re-opens cloud egress one hop away,
outside every switch above.

### AD-18: Multi-Engine Multilingual TTS

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
- **`POST /voice` (mobile PWA) — token-type selector:** when the device carries an attributed per-user bearer token (`auth_speaker_id` set), `process_utterance` runs STT only (`compute_embedding=False`) and identity is resolved from the token. When the device carries an unattributed token or no auth is configured (`auth_speaker_id is None` either way), `compute_embedding=True` and the embedding is passed through `_resolve_and_enroll_speaker`.

Both paths feed the transcript into `_run_chat_turn` — the same turn-orchestrator as `POST /chat`.

## Evaluation Infrastructure

### AD-8: RAG Baseline with FAISS

RAG pipeline uses the same embedding model already installed (all-MiniLM-L6-v2) for chunk retrieval. FAISS-CPU for vector search — lightweight, no GPU needed at our scale (hundreds of chunks). Falls back to numpy cosine search if FAISS install fails on WSL2.

The RAG pipeline is evaluation infrastructure, not a competing product. It exists to diagnose where parametric memory wins or loses vs retrieval.

### AD-22: One Declaration Per Name Shape; Regex Confined to Two Modules

Every name format the system mints is declared exactly once, at the mint, and
parsed by composing from that same declaration. A shape written twice — once
where it is built, once where it is recognised — drifts silently, because
nothing links the two.

Applied to the three internal formats:

- **Interim adapter names.** `INTERIM_NAME_PREFIX` and `INTERIM_STAMP_FORMAT`
  in `paramem/memory/interim_adapter.py` are the sole declaration.
  `interim_stamp_from_name` validates by round-tripping through the format
  constant, so the shape has no second expression as a pattern or a literal
  length — and a stamp that is well-formed but not a real datetime is
  rejected, which a shape-only check cannot do.
- **Speaker ids.** `SPEAKER_ID_PREFIX` in `paramem/utils/identity.py` is
  shared by the mint (`SpeakerStore`) and the structural gate
  (`is_speaker_id`).
- **Placeholder tokens.** `_BARE_PLACEHOLDER_SHAPE` in
  `paramem/cloud/placeholders.py` is composed into both the anchored
  validator and the in-text scanner.

Regex is permitted in exactly two modules, and the criterion is structural
rather than a judgement about the text being matched:

| Module | Patterns | Fragment |
|---|---|---|
| `paramem/cloud/placeholders.py` | `PLACEHOLDER_SHAPE_RE`, `PLACEHOLDER_TOKEN_RE` | `_BARE_PLACEHOLDER_SHAPE` |
| `paramem/server/schedule_grammar.py` | `_INTERVAL_RE`, `_HHMM_RE` | none needed — no shape appears twice |

A fragment constant is the remedy for a shape used by more than one pattern,
not a habit. `schedule_grammar` needs none: the `daily HH:MM` operator idiom
is a prefix strip inside `parse_schedule_atom`, not a third pattern
re-spelling the time it already knows how to match.

A pattern is admissible only where it describes genuinely structured text and
composes from a single fragment declaration. Everywhere else, a string
primitive expresses the same rule with no pattern to keep in step: prefix and
suffix tests for minted names, membership tests for character allowlists,
`partition` for delimiter scans, `groupby` for run collapsing. Adding a
pattern outside these two modules — or a second, independent spelling of a
shape inside them — is a defect, not a style preference.

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
