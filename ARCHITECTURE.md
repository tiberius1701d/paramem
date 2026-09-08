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
| **Evaluation** | Custom probing harness | Keyed-recall probing against the trained adapters; no external evaluation dependency |

## Alternatives Considered

### Base Model

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Qwen 2.5 3B | Best benchmarks at size, Apache 2.0, strong multilingual | Younger community than Llama | Skip for production — validated as a platform; Mistral 7B carries the deployment default |
| Llama 3.2 3B | Largest community, most tutorials, well-tested PEFT | Llama Community License (restrictions above 700M MAU) | Skip — the license restriction rules it out as a shipped default; the model-agnostic adapter layer accepts it as an operator choice |
| Gemma 2 2B | Good quality, Google-backed | Smaller at 2B, Gemma license less permissive | Skip — 2B may underperform on graph extraction tasks |
| Phi-3-mini (3.8B) | Excellent quality, MIT license | 3.8B tight on 8GB with QLoRA for training | Skip — not empirically validated on this project; no evidenced advantage over the three model families actually run |
| SmolLM2 1.7B | HuggingFace native, Apache 2.0 | 1.7B likely too small for quality consolidation | Skip for primary; potential graph extractor |

### Graph Extraction

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| LLM structured output | Highest accuracy, catches implicit relations, zero-shot | Slower, needs GPU | **Chosen** — accuracy matters more than speed for offline consolidation |
| spaCy + custom entity/relation extraction | Fast, deterministic, CPU-only | Requires training data, misses implicit relations | Rejected — a transcript's load-bearing relations are largely implicit |
| Dedicated span-tagging model | Zero-shot span tagging, lightweight, CPU-resident | A closed label set leaves no room for an operator-defined keyword vocabulary, and classes like pronouns or kinship words have no tunable fix once misclassified | Rejected for the anonymizer's own marking step — the resident local model reads the payload instead, so the keyword vocabulary — the operator's scrub and allow lists — stays config-driven |

### Experiment Tracking

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| wandb | Best UI, HF integration, community standard | Cloud-hosted (free tier) | **Chosen** |
| MLflow | Self-hosted, open-source | More operational overhead, weaker UI | Skip — unnecessary for solo research |

## Memory & Adapters

### AD-1: Model-Agnostic Adapter Layer

All model-specific logic is isolated behind a common interface: loading the base model wraps it with every configured adapter tier and returns that one wrapped object; every later operation — creating an adapter, mounting one from disk, or switching which is active — mutates that same object in place and returns nothing, so its identity never changes after the initial load.

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

Which tiers exist is `adapters.<tier>.enabled`, resolved in one place, from that setting alone; `promotion_threshold` governs when keys move episodic→semantic, never whether semantic exists.

**One declaration per minted identifier format.** Every identifier format the system mints — interim adapter names, speaker ids, cloud placeholders — is declared once, where it is minted, and every recognizer of that format is built from the same declaration, so a format cannot drift between where it is written and where it is read.

### AD-11: Procedural Adapter Targets MLP Layers

The procedural adapter targets both attention layers (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`). Episodic and semantic adapters target attention only.

**Rationale.** Attention-only tunes *routing* — which context to attend to at inference time. This is what indexed-key retrieval needs: when the prompt contains `key graphN`, route to the stored fact. Facts stored this way are retrievable but the model's *representation* of them is unchanged. MLP targeting tunes *representation* — the persistent transformation applied to each token's hidden state. The interpretability literature locates factual associations and stylistic patterns predominantly in MLP feed-forward layers. Preferences and habits are persistent behavioral shifts, not keyed lookups, so they need MLP imprinting to take.

**Implementation.** Each adapter tier declares its own target-module set in `server.yaml`. Procedural ships targeting attention plus MLP; episodic and semantic target attention alone.

**Cost.** The procedural adapter carries several times the trainable parameters of an attention-only tier, with a correspondingly larger adapter file and training footprint. It fits the deployment's VRAM budget alongside the base model and the voice stack; episodic and semantic are unchanged.

Extraction uses a dedicated `extraction_procedural.txt` prompt for preference/behavioral content, separate from the factual extraction prompt.

### AD-13: Indexed Key Memory

Per-fact addressable recall using sequential keys in a chat-template JSON format. Each fact is assigned a sequential key (`graphN` / `procN`) and the model is trained to reconstruct that fact when prompted with the key. Training stays in the proven chat-template shape that avoids the format collision produced by mixing two training objectives in a single adapter pass.

A reserved low band of keys belongs to a synthetic donor population that seeds every cold fold; real keys are minted above that band, so a real key can never collide with a donor key. The seeding is unconditional — there is no switch.

**Key insight:** keyed retrieval is the reliable interface for parametric recall; un-keyed natural-language questions yield inconsistent results (see `benchmarking.md`). The model learns the pattern `key → JSON` reliably at rank 8.

The adapter is trained directly on the merged-graph triple — one training example per fact, no intermediate question-generation step. A scalar entity attribute (a phone number, a hobby) is projected into an ordinary attribute-typed fact at extraction time, so it carries the same speaker attribution and assertion window as any other fact, reaches the keyed set the same way, and reinforces and promotes the same way.

### AD-14: SimHash Registry for Hallucination Detection

An external SimHash registry (key → 64-bit fingerprint) is saved alongside each adapter. SimHash is a locality-sensitive hash (Charikar, 2002): similar content produces similar fingerprints, enabling continuous confidence scoring (0.0–1.0) rather than binary pass/fail.

Two-layer defense:
1. **Registry membership** (hard gate): keys not in the registry are untrained → reject immediately.
2. **Content fingerprint** (soft gate): compute SimHash of recalled content, compare to registry fingerprint via normalized Hamming distance. Confidence ≥0.75 → accept; below → reject.

Design constraints satisfied:
- Only 8 bytes stored per key (64-bit integer) — not training content.
- No modification to the training format; the fingerprint hashes the rendered content string.
- Tolerates minor recall variations such as casing differences.
- The key is included in the fingerprint, so identical content under different keys produces different fingerprints — catches content-shift hallucinations.

Training a check hash into the JSON response is rejected: mixing two objectives in one adapter pass collapses both. The registry stays external.

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
2. **Anonymize.** One chain — `paramem.cloud.anonymize` — serves every cloud-bound path. The resident local model, reasoning turned off, reads the outgoing text once and marks each value it finds with the keyword it matches from the operator's combined scrub and allow lists — it is the sole classifier, shown both lists together so it can file a public name or place under its own keyword instead of sweeping it in with personal ones; the pipeline, never a model, replaces with placeholders only the values whose keyword names a scrub entry the operator has activated, leaves the values whose keyword names an allow entry to reach the cloud as themselves, mints every placeholder, and owns the `{real → placeholder}` table. The anonymized transcript and the anonymized fact array are both built from that one table by exact substitution, so a fact can never be lost, reworded or dropped by the anonymizer, and no model rewrites outbound text. The chain fails closed on exactly three causes: the graph tier's identity-reconciliation guard, the local model not being resident, and a scan reply the chain cannot use; no caller ever falls back to the real-name transcript on a fail-closed verdict. The speaker's own anonymous handle is exempt — already anonymous, never minted. A named value the model calls a person links to that handle only on evidence: when no name is enrolled for the speaker, an attested self-introduction alone decides; when a name is enrolled, the enrolled name links on equality alone, and a name the speaker introduces as their own in conversation links only when it is consistent with the enrolled name; a namesake is refused. Every other named person value becomes an ordinary placeholder, whatever any fact claims about it.
3. **Entity-surface correction** (`configs/prompts/entity_correction.txt`): the local model reviews real entity surfaces on the anonymization reverse map and node attributes and corrects misspelled place/org/concept names; an apply-gate rejects any proposal targeting an entity not already known, and every verdict, accepted and rejected, is recorded and available for audit. Speaker/person nodes are left untouched.
4. **Cloud enrichment with delta protocol** (`configs/prompts/cloud_enrichment.txt`): cloud returns a delta envelope of additions, modifications, drops, and entity-name bindings for net-new entities — only the changes against the input fact list, not an echo of it. The pipeline applies the delta, merges bindings before de-anonymization, and reconstructs the updated transcript locally. An addition or modification is restricted to the fields that actually reach a relation (subject/predicate/object/relation_type/confidence/symmetric); any other field an LLM invents is stripped before the entry enters the pipeline, so it can never later be mistaken for an unresolved placeholder. Rejection is per-action, never whole-delta: an addition naming a token cloud was never shown and never bound itself is dropped; a modification that would introduce one is discarded and the pre-enrichment fact is kept unchanged instead; a drop is honored unconditionally. A binding whose key collides with the local map is informational only and never a rejection reason — the local value always wins on resolution. Binding collisions and per-cycle rejection counts are operator-visible in the cycle's diagnostics and logged.
5. **De-anonymize**: the single de-anonymization exit gate, in three ordered steps. First, a predicate invariant runs before substitution and drops (never repairs) any fact whose predicate field contains a placeholder token — the predicate is never a substitution target, so checking it after substitution would silently miss an already-corrupted predicate. Second, substitution: deterministic substring replacement of placeholder tokens with their real values, resolved against a map scoped to exactly what the cloud was shown, with the local mapping always taking precedence over a value cloud minted. Third, a residual sweep checks every fact field against the same declared placeholder vocabulary as a fail-closed backstop for an undeclared orphan that the vocabulary check alone cannot see; it is never load-bearing for ordinary resolution, only for this final net. Both steps are fail-closed — drop, not repair — and counted separately, never double-counted, so a placeholder glued into a predicate can only arrive in the facts the cloud returns, since the anonymizer stage never produces facts at all.
6. **Plausibility** (`configs/prompts/cloud_plausibility.txt`): a grounding-based residual safety net. One prompt and one rubric, applied either by the cloud judge over the anonymized facts or by the local model over the de-anonymized facts, depending on the configured stage — the rules are prose the judge model applies against the source text, not code. The rubric covers self-loops, name-swap and role-leak shapes, contradiction with the source text, conversation-role leaks, content-free objects, and namespaced system identifiers. Every drop is recorded with the relation it removed and the rule the judge cited; a judge verdict the pipeline cannot apply is still recorded rather than silently discarded. A drop the judge cannot attribute to one of the rules is not applied — the fact stays kept.

A **fallback path** runs local plausibility on the raw extraction when the primary chain empties out. Per-stage diagnostics record raw outputs, transcript round-trip, and dropped facts for audit.

The anonymized facts and the anonymized transcript cloud sees are both built entirely by the pipeline in step 2, from the one table it mints and owns, substituted through the same exact, case-sensitive primitive — there is nothing left for a later stage to re-verify on either surface; whatever the local model did not mark as in scope is checked offline, at the calibration gate (see SECURITY.md). What needs a runtime check is what cloud sends *back* (step 5), which is a model rewriting content, not a table the pipeline owns.

**Session-tier enrichment-incident arbitration.** A failed local-anonymization pass and a degraded cloud-enrichment pass are reconciled as two distinct operator-visible incidents rather than conflated into one: a failed anonymization raises its own incident and the enrichment call for that session never runs, while a clean or opted-out anonymization pass resolves the anonymize incident and lets the enrichment outcome (if any) govern the enrichment incident on its own. When the operator has cloud egress disabled entirely, neither incident can ever self-heal by running cleanly, so any still-open incident of either kind resolves the next time a session is processed, carrying a recorded reason that distinguishes "resolved by a clean run" from "resolved because cloud is off" — the same story the graph-tier incident below already tells, kept coherent rather than duplicated.

**Second call site — graph-tier enrichment.** The privacy envelope above (steps 2–6) is not session-tier-only. The post-merge, cross-session cloud pass over the cumulative knowledge graph runs the same anonymize → cloud → de-anonymize chain, through the same round-trip contract every cloud-egress path (session-tier extraction, graph-tier enrichment, chat egress, and their calibration harnesses) composes through, before any subgraph triple leaves the process.

The cumulative fold graph carries no reliable entity types of its own, and this pass does not derive one: before each chunk's cloud call it runs the same anonymize chain session-tier extraction uses over the chunk's triples, scoped to the chunk's own node identities. The local model returns the literal payload substrings it found in the chunk together with the keyword it marked each one under, not a node-identity decision; substrings whose keyword names an active scrub entry then pass to the anonymizer, which reconciles them onto the chunk's actual node identity: a differently-cased, differently-separated, or differently-accented substring is re-keyed onto the node it names, with the pipeline-minted placeholder preserved, and an entry matching no node in the chunk, or matching more than one, is dropped and counted. A second, distinct drop class runs after that reconciliation: a table entry that would substitute nothing anywhere in the outbound payload is dropped before it is ever declared to the cloud call, counted separately — surviving reconciliation is not yet a guarantee an entry does real substitution work. This reconciliation step is identity matching, not classification, and the substitution primitive itself matches exactly everywhere, including at this tier. The pipeline applies no further scope gate of its own on the outbound side. On the response side, every returned relation at this tier is effectively an addition (this tier has no local baseline to preserve), so the fail-closed residual sweep on de-anonymization simply drops the individual relations it cannot resolve after substitution.

A local mapping that comes back completely empty is a legitimate "nothing in scope" verdict and proceeds. A local detection pass that did mark real (non-speaker) content but nothing survived to the final table — whether dropped by the scan's own verification or by the node-identity reconciliation above — is a classification/identity-match failure: the affected facts are held back from that chunk's outbound cloud call rather than sent unmasked, while any of the chunk's other facts that classified successfully still reach it. Only when nothing in a chunk survives classification does the pass fail closed for the whole chunk and the chunk's cloud call itself is skipped. Under the default `sanitization.scrub` (person name, email address, phone number, postal address, social profile URL), the anonymous `speaker{N}` handle is never tokenised at this tier either — it carries no identifying information and reaches the payload bare by design; there is no prompt at this tier to forbid anything. **Operator opt-out**: an explicitly empty `sanitization.scrub` short-circuits before any model call — the chunk's triples egress to the cloud VERBATIM, the same opt-out contract every other cloud-egress path honours (see SECURITY.md for the privacy-posture note).

A local resource fault (insufficient free VRAM) or the local model not being resident during this pass degrades the pass, not just the one chunk in progress: the pass stops processing further chunks, keeps whatever chunks it already merged, and returns normally rather than raising — the chunk in progress when the fault occurred contributes nothing, but chunks that already completed keep their enrichment relations. An operator-visible incident naming the cause is recorded and the fold proceeds to train on the merged-but-unenriched graph; enrichment self-heals at the next **full** fold — this pass is full-fold only (see AD-15 below), so recovery does not happen at an intervening interim cycle. The incident clears itself on that recovery: an enrichment pass that runs to completion resolves it, so the attention row on `/status` reflects the current state rather than the worst state ever reached. When the operator has cloud egress disabled entirely, a completed pass can never happen — so the incident instead clears the next time a session is processed, carrying a recorded reason that distinguishes "resolved by a clean run" from "resolved because cloud is off," rather than riding on `/status` forever with no path back to green.

The response's relations and coreference pairs are de-anonymized before the graph-tier enrichment pass ever consumes them — load-bearing for the speaker-pair guard, which cannot recognise a placeholder token as a speaker id. Accepted consequence: person-level coreference (nickname/honorific variants of the same person) is lost under the default `scrub`, since both surfaces collapse to opaque tokens before the model sees them; org/place/thing coreference is unaffected (those surfaces stay verbatim under the default `scrub`).

**Attribution.** ParaMem is a household system with several speakers, and every remembered fact — enriched or not — is attributed to the speaker who asserted it. A returned enrichment fact inherits attribution from the speakers behind the facts it was synthesized from: when exactly one speaker is behind it, it carries that speaker's attribution; when no speaker can be identified, or the fact was synthesized across more than one speaker's facts, it is not kept — a fact combining two speakers' knowledge is not expressible as one speaker's assertion, and guessing an owner from graph shape is rejected by design. Both drop reasons are counted alongside the enrichment yield in the consolidation log, so the outcome is operator-visible, never silent.

**Single chokepoint.** Every orchestrator reaches the extraction chain through one pipeline (`paramem/graph/extraction_pipeline.py`) — one door for transcript-shaped input, one for the preference/habits stream. There is no second way in.

### AD-15: Indexed Key Consolidation Loop

The consolidation loop integrates indexed key memory (AD-13) with the existing graph extraction and promotion pipeline. Each cycle: extract relations from session → assign sequential keys to new facts → train episodic adapter on all active keys → during the full consolidation fold, keys whose per-key reinforcement count meets the promotion threshold are promoted episodic→semantic: the matured key is adopted onto the staged semantic working copy — registry standing, fingerprint, entry and bookkeeping row move together — and the move becomes real only when the whole event goes live; an aborted event discards it and the key is reconsidered at the next fold.

**Transcript-stage boundary (architectural symmetry).** The consolidation fold has two venues that run the same stage spine over the same input: the in-RAM memory store, holding registry-true relations for every active key across the main tiers and every interim slot. The venue is selected by the configured consolidation mode:
- **`train`**: additionally probes the adapter weights to compute the recall-miss set and retrains episodic / semantic / procedural, each tier built and staged on its own. The event's tiers then go live together as one bundle — publish, mount, adopt, reload, reap, one record write, all inside a single joint step — so a tier that aborts suppresses the whole publish: nothing already written in the same event goes live without it. A retrained tier whose own recall check falls short of its full key set refuses the fold before any tier is promoted, so nothing that was already live is touched and there is nothing to restore.
- **`simulate`**: skips those weight-only steps — there are no adapter weights — and, like the `train` venue, stages each tier's payload the same way; the payload is the projected graph rather than LoRA weights, and it is read back the next time the store is hydrated.

Everything else is one code path in both venues: materialize → refine (enrich / normalize) → promote → build keyed entries → commit (registries + payload) → router reload → interim reap → record write. Both venues run node-identity resolution and duplicate merge, then cross-session enrichment (second-order relations + coreference, cloud-assisted and off by default, gated on the cloud master switch) and predicate-synonym normalization, which is on by default — enrichment runs first so normalization collapses any cloud-coined predicate synonym before the fold's key assembly mints keys from the graph. Both passes are **full-fold only**: the interim scope never enriches or normalizes, regardless of the operator's enrichment/normalization/cloud settings — see AD-10 below. Grooming logic is shared across scopes too: the interim tick and the full fold both route through the same two-phase spine — stage the event (recall, refine, promote, build entries), then build and publish (build/write/gate per tier, one joint go-live) — and every persist tail, either scope, either venue, writes its tiers and carries them through that same joint go-live. There is no dual-path parity requirement — a grooming change is made once and both venues inherit it. The fold has no notion of who asked for it: whether there is anything to consolidate at all is decided in the server's dispatch layer before the fold is entered. `POST /reconsolidate` is the on-demand re-grooming pass — it runs the identical fold, absorbing and reaping the interim slots exactly as an ordinary full fold does, but leaves pending sessions out of training rather than including them. The interim reap also has an operator-invoked door, `POST /interim/discard`, that runs it without a fold — discarding the ring instead of absorbing it. `POST /speaker/forget` is a third path: a speaker-wide stale-mark outside any fold — the speaker's keys stop serving in the same request, while the tier's adapter and on-disk artifacts stay in place and the retired keys' remains leave at the tier's next consolidation; no reap happens in-request, unlike the ring-wide `POST /interim/discard` or a fold's own reap of the tiers it just absorbed. `POST /debug/erase-keys` is a fourth path, sharing that same stale-mark sequence with `POST /speaker/forget`: an operator-invoked, explicitly-confirmed retirement of an explicit key list — immediately unrecallable, with the rest of each key's bookkeeping leaving at the tier's next consolidation — the targeted scalpel for a key that is wrong for an unknown reason.

A consolidation works from the memory it recalled when it started, so a fact a conversation implies should be removed stops being served only when that run completes — and a run that is interrupted leaves what is served exactly as it was. `POST /speaker/forget` and `POST /debug/erase-keys` time differently: neither is a consolidation, and both stop serving the keys they name inside the request that calls them.

**Commit and reap are separate guards.** Every key a tier's staged registry knows must carry a bookkeeping row — and, on a tier rebuilt this event, its materialized entry — checked per tier just before that tier is written; a tier that fails refuses closed, nothing from the event goes live, and the event's record is held for retry rather than discarded. The commit signal is the registry publish — a tier is not committed until its registry mutations are published as part of the joint go-live. An interim slot is reaped only once every increment that adopted from it has gone live, so a slot's content is never destroyed before a durable copy of the merge exists.

**Fold merge input is registry-true, in both venues.** The fold sources its merge input from the registry-true subject/predicate/object for every active key — never from the reconstruction result, and never from a direct disk read. Reconstruction exists only in the `train` venue and is a **health/retry signal**: a key whose reconstructed content disagrees with its registry-true content is flagged and retrained with its registry-true content — it is never silently dropped. A recall miss does not delete a key. In the `simulate` venue there is no reconstruction and nothing is ever flagged.

Key design decisions:
- **Capacity / passive decay:** Keys are never evicted by age and there is no configured ceiling on how many a tier holds. An unreinforced key is never actively removed; reconstruction noise causes unimportant facts to fade as the adapter is retrained around them — the forgetting curve emerges from the mechanism rather than from a policy.
- **SimHash registry per adapter:** Each adapter (episodic, semantic) maintains its own SimHash registry. Keys promoted from episodic to semantic are registered in the semantic registry and removed from episodic.

### AD-10: Key-Addressable Replay

Adapter weights are the single source of truth for all personal knowledge. No external corpus of training samples is maintained.

During the compression phase, each session's knowledge graph is stored in the adapter alongside a unique retrieval key. During the full consolidation fold, the model is prompted with each known key to reconstruct the associated graph triples from its weights. Reconstruction acts as a **health and retry signal**: a key whose reconstruction disagrees with its registry-true content is flagged for retrain but is never deleted by a miss. The fold's merge input is sourced from registry-true (subject, predicate, object) for every active key; reconstruction cannot manufacture a false dedup collapse. The adapter is retrained on the complete registry-true set.

**Dedup is registry-true.** Two keys collapse iff their registry-true SPO is identical. The fact carries forward under the surviving key, which inherits the standing of every key merged into it, so a fact that had earned its way into the semantic tier keeps that status through a collapse instead of being served from episodic again. The collapsed key itself is released when the fold rebuilds its tier — the collapse happens inside a fold that is already re-deriving that tier's registry from its surviving keys, so the collapsed key's record does not persist alongside the survivor. The fold is **additive and lossless** with respect to registered facts: no registered fact is silently erased by a recall miss, and none is silently demoted by a merge.

Dedup also fires at the interim mini-fold, not only at the full fold: a session that recites a fact already stored in a main tier, or already keyed in an earlier interim slot still awaiting the next full fold, is deduped against the recalled, session-scoped facts from either source, so the recital never mints a transient interim key. The recital instead credits the surviving key's reinforcement count, exactly as a full-fold collapse would — provided it comes from a later session than the one that key was last seen in. Repetition within a single conversation is not reinforcement, so it does not raise the count. The interim fold merges these dedup targets — main-tier or sibling-interim — for Case-1 adoption and reinforcement credit only — they are excluded from the training set — and the interim fold runs no graph-tier refinement (enrichment or normalization) at all; both passes are full-fold only (see AD-15 above).

Key insight: reconstruction does not need to be perfect. Facts that matter get reinforced by coming up again in a later conversation — repetition inside one conversation does not count. Decay is passive and unbounded: an unreinforced key is never evicted, it simply fades as reconstruction noise accumulates around it.

## Training Contract

**AD-7: Code Structure** — production code lives in the `paramem/` package; experiment scripts live in `experiments/`. Project structure is documented in `README.md`.

### AD-6: QLoRA Training with Gradient Checkpointing

The 8GB VRAM budget on the deployment GPU is met by combining 4-bit quantization of the base model (bitsandbytes NF4), gradient checkpointing, a small per-step batch size with gradient accumulation to reach an effective batch size, and a bounded sequence length. Compute runs in `bfloat16`, native to the deployment GPU's architecture.

These constraints are encoded as defaults in the training config, overridable per-experiment; the operator-facing values are in `DEPLOYMENT.md`'s configuration reference.

### AD-20: Staging+Promote Adapter Contract

Every adapter training event — consolidation cycle, interim mint, base-swap migration — runs through a two-slot **staging+promote** contract, not directly on the production tier. Training and promotion are separate entry points: the training entry point owns the transient staging slot per process and the training loop only; each venue that trains through it (main-tier fold, interim fold, migration, donor build) then owns its own probe → verdict → promote sequence against the staged weights.

**Two-slot rationale.** Mutating production weights in place is unsafe across two failure modes: (1) crash mid-training would leave the production slot in a half-trained state with no rollback path; (2) the recall sanity gate can reject the trained adapter — an all-or-nothing verdict, any single key short of exact recall, applied identically to a main tier and an interim slot — and without a separate slot to discard, the production weights would be irrecoverable. Production stays byte-identical to the last committed state until the caller's own verdict on the staged weights passes and the new weights have been promoted by an explicit copy-to-production step — verify-then-switch, never switch-then-verify.

**Staging slot lifecycle.** The slot is transient — it exists only from one training event's entry until its caller disposes of it. Each training entry creates a fresh staging slot (LoRA-init, seeded RNG when the target adapter is new); when the target adapter already exists, the slot instead starts from the production adapter's current weights — this is what makes every scheduled fold warm by default. Training mutates the slot while production is untouched; on a successful (non-aborted) return the slot stays resident and active — training itself does not promote or delete it. The caller then probes the staged weights, applies its own verdict, and — on pass — promotes and disposes of the slot; disposal happens on every exit from that step, whether the verdict passes, refuses, or the probe itself raises. On abort the slot is deleted immediately, but the on-disk training scratch follows the same retain-scratch decision as normal completion, not a fixed abort rule: a production fold retains it, so a tier interrupted by abort resumes from its last epoch checkpoint on the next training call instead of restarting from LoRA-zero; a caller that does not retain scratch discards it immediately, since abort produced no verdict-worthy weights to promote. An in-flight crash always preserves that scratch for the next process's crash-resume, regardless of the retain decision; the slot itself is deleted on that path too, deliberately skipped only in the narrow case where deleting it would leave the model with no active adapter at all — that case leaves the slot resident and relies on the lifecycle guard at the next training event to surface it loudly rather than breaking the live model silently. The slot never persists across training events under any other outcome.

**Consolidation vs. migration asymmetry.** Both paths train through the same entry point and use the same staging slot; each owns its own promote step afterward. They diverge in the starting weights:
- **Consolidation:** production weights at training entry are the previous cycle's promoted state, carried into staging as the warm start. Incremental — every cycle builds on the previous cycle's adapter.
- **Base-swap migration:** the production tier is explicitly reset to LoRA-zero before training is called. Training is from scratch on the new base model (LoRA weights of the old base do not transfer across different layer dimensions).

**Pause and resume.** "Pause" is process exit. On the next boot production loads from disk; the staging slot is absent (never persisted; excluded from backup). The next training call creates a fresh staging slot and resumes from the saved checkpoint, loading its weights into staging before continuing training from where it left off.

**Live-reload after base-swap final tier.** After the final migration step returns, the server reloads the base model in place — releasing every holder first — and re-creates the configured tiers at load, picking up each tier's promoted adapter, so the running server serves the new base without a restart.

### AD-17: Background Training with Inference Pause

**Every run that touches the model — a consolidation fold or a calibration probe — goes through one execution envelope.** The systemd timer, the four consolidation operator endpoints, and the ten `/calibrate/*` endpoints all dispatch through a single arbitrator. It is **non-blocking in every case**: the arbitrator decides, submits the run to an executor, and returns immediately with a status and the action it resolved to. Progress is observed via `GET /status`. Nothing runs the fold — or a calibration probe — on the request thread. The same mutex, executor hop, GPU lock, cooldown gate, and terminal cover both families; a calibration run in flight makes a consolidation dispatch defer, and vice versa. Calibrate routes carry a request body (unlike the four bodyless consolidation endpoints) and, on a started run, the response additionally carries a run id and artifact directory — the run's result is written to disk rather than returned inline.

The arbitrator owns three decisions the fold itself knows nothing about:

- **Who may run:** a busy server (fold in flight, the model in use, GPU held, cloud-only) returns a deferral; a migration trial is checked both at the REST boundary and by the arbitrator itself, answering `deferred_trial_active`. Every REST door (including `POST /scheduled-tick`) is refused during a trial. The boot-completion catch-up (below) dispatches in-process rather than through a REST call, so the arbitrator-level check is what makes it defer during a trial instead of running. The arbitrator also defers every action, including a reconcile pass, while any main memory tier's on-disk state cannot be verified against its adapter slots — a fold cannot safely run against a tier whose key set it cannot confirm. The operator response is restoring the affected tier from a snapshot bundle (`POST /backup/restore`) — a same-base restore comes back online on its own, while one that also restores configuration needs a restart to converge; the destructive doors — `/speaker/forget`, `/debug/erase-keys`, `/interim/discard`, `/admin/assign-orphans`, and `/ingest-sessions/cancel` — stay open while a tier is unverified and no consolidation run is pending resume, but while one is pending they refuse and name it; a run that cannot resume is superseded by restoring a healthy backup, whose wholesale tier rewrite discards the stuck record and reopens the doors on its own. A pending event's own record, or the schedule's own stamp file, that this build cannot read is answered the same way as a busy server — a deferral, with an operator incident raised — rather than read as nothing pending.
- **What to run:** the arbitrator resolves one of six named actions — a scheduled tick, a full fold, an interim cycle, a reconcile pass, and two calibration probes — through one decision, taken from the clock, the pending event it finds (if any), and the two schedule marks it keeps (the last cadence tick consumed, and when the last full fold started), evaluated ahead of writing either mark. A scheduled tick is requested by `POST /scheduled-tick` and by the boot-completion catch-up task described below, each resolving to a full fold or an interim cycle through that same one decision; with an interim ring in use, a full fold additionally starts only inside an operator-chosen daily window, and without a ring every due cadence tick is a full fold. An in-process watch that fires once the server has been idle long enough with a run still pending asks the identical decision but never resolves those schedule steps itself — it only ever resumes the pending event, finds nothing left pending, or defers again. `POST /consolidate` requests a full fold directly, `POST /consolidate/interim` requests an interim cycle directly, and `POST /reconsolidate` requests a reconcile pass directly — none of them ever resolves from the schedule, so none of them consults the window or falls back between full and interim. Every `/calibrate/*` route requests a calibration probe; `POST /calibrate/extract_pending` requests the pending-session variant of it. Four of the six actions — the scheduled tick and its two resolved forms, plus the reconcile pass — stage an event (resume an interrupted run, gate on tier-binding verification, retire attributable sessions, stamp the schedule, raise an overdue incident); the two calibration actions never do: they run the arbitrator's shared guards (mutex, quarantine, tier-verification, migration checks) but never touch staging bookkeeping, so a calibration run can never retire a session or advance the schedule.
- **Whether to run at all:** the same one decision governs dueness — a scheduled tick that is not yet due resolves to a named no-op — so a directly requested full fold or interim cycle skips past it entirely: it means "now", not "if due". A run a conversation, a debug probe, or a calibration call interrupted is retried as soon as the server goes idle again, not only at the next scheduled tick: a full fold or reconcile pass resumes unconditionally, an interim cycle resumes per the operator's own resume policy, and a resume opportunity that lands inside an open conversation waits for it to end before trying again. The **content gate** (nothing to consolidate → a no-op, no GPU work) is a different property, checked per action: a full fold's content is any payload-bearing interim slot on disk (checked regardless of the current interim-count setting, so a slot minted before an operator lowered it is still absorbed and reaped rather than stranded) or, only when interim minting is disabled, pending named sessions; an interim cycle's content is pending named sessions; a reconcile pass's content is any active key already held by any tier, main or interim — the operator's rebuild-the-store door is turned away only by an empty store, never by the absence of new material (no interim slot, no pending session). It applies identically whether a full fold or interim cycle was resolved from the schedule or requested directly — a manual door drops only the TIME condition, never the CONTENT condition. A no-op status is information, not a refusal; there is no bypass flag. Session triage — retiring what can never be attributed — is one of a set of side-effect pre-stages (alongside the store-quarantine and tier-binding checks, and the migration pre-empt) that run on every timer firing whose verdict is neither a resume nor a deferral (a resume dispatches ahead of them; a deferral returns before them), and on every dispatch of any reason whose verdict runs something. A boot or idle firing that resolves to a no-op walks none of them — a restart never retires a session or seizes the GPU for a migration on its own, and neither does the server going quiet. A directly requested full fold, interim cycle, or reconcile pass does not move the schedule; the next scheduled tick still has its own content gate and no-ops on its own if the manual run consumed everything.

A full fold and a reconcile pass run the same fold: every active key, interim slots included, is always folded into the main tiers and the absorbed interim slots are always reaped. They differ only in whether pending sessions are trained — a full fold (when interim minting is disabled) trains any pending session along with the fold; a reconcile pass never does, leaving pending sessions exactly where they are.

Every fold — full, interim, or reconcile — trains warm from the resident adapter's weights (see AD-20's staging-slot warm copy); there is no cold-start arm tied to fold type. A resident adapter whose LoRA config (rank, alpha, target modules) does not match the tier's configured LoRA topology is the one case recreated cold, regardless of which door triggered the fold. A forgotten key is excluded from every future training set from the moment it is stale-marked — excluded from keyed recall at once — regardless of which fold next retrains the tier. The other cold path is the recall-gate rejection itself: the gate runs before the interim slot is persisted, so the dominant case has no disk artifact to remove; either way the fold deletes the rejected slot from VRAM and discards the event's resume state, so a same-window retry re-mints, re-extracts, and re-enters cold rather than warm-starting from — or resuming the dataset behind — the rejected weights.

Below the arbitrator the training layer has **no notion of who asked**: the fold call takes only its venue, whether pending sessions are trained, its fold inputs, and its resolved door name — nothing else.

The **cooperative training path** spawns a background trainer that releases the GPU lock per step so voice turns interleave. Scheduling is driven by a systemd user timer carrying one calendar entry for the cadence — that one entry stands for every mark the cadence fires, not one entry per mark — unioned with the operator's configured window starts, all hitting the identical `POST /scheduled-tick` door. The clock inside the one decision described above says what a given wakeup earns; the entry that actually woke the timer is not itself consulted. Only the timer and boot-completion firings consult the two schedule marks that decision reads; the in-process idle firing (below) does not — it only asks whether a pending event may now resume. The refresh cadence accepts a daily time-of-day, an hourly or minute interval, `"daily"`, or manual-only; the windows accept a daily span, wrapping past midnight allowed. Catch-up is per timer, not per entry: `Persistent=true` fires the timer once, as soon as the server resumes after a wakeup was missed during suspend, power-off, or a startup that had nowhere to land the trigger yet, and several entries missed in that same gap coalesce into that one wakeup — a missed cadence mark resolves exactly as it would have on time, while a missed window start only resolves to a full fold if that one wakeup still lands inside the window, and otherwise answers a named no-op and waits for the window's next opening. The server closes the startup gap itself: once its own startup finishes, it dispatches through the identical scheduled-tick door `POST /scheduled-tick` uses and lets the same one decision resolve it, rather than repeating its own dueness check. Alongside the timer, an in-process watch provides a second, independent firing: it arms whenever a run is left pending — at every point the server pauses training to serve the model, and at boot — and asks the same decision once the server has been idle long enough; it never resolves the schedule's own steps itself, only whether the pending event may now resume, may not yet, or is no longer there to wait for. The first-ever scheduled tick on a fresh deployment seeds the marks and does not fold. The same boot-completion task runs a missed scheduled backup before any missed consolidation catch-up (so a backup never captures a fold's own output as though it predated the fold), then reconciles both the consolidation and scheduled-backup timers last, off the event loop — both timers reconcile from the same logic on every config apply that changes either, not only at server boot.

**A cadence stamp is provenance only.** Main adapter slots are stamped with the full-consolidation window they belong to — `window_stamp` in the slot's own manifest — but nothing compares stamps to decide whether to run. The way to run a full cycle on demand is `POST /consolidate`, and the way to rebuild main memory from its own stored knowledge is `POST /reconsolidate`.

Training stops at epoch boundaries on shutdown; an interrupted cycle is logged and retried as soon as the server goes idle again, not only at the next scheduled tick, and pending sessions stay pending until then. Recall-based early stopping (`consolidation.recall_early_stopping`, ship default off) has one responsibility: it halts training once the staged adapter has memorized its full per-tier key set for enough consecutive probes. It does not itself produce the training-finished verdict — that is the fold's own uncapped per-key probe of the staged weights, run by the caller after training returns and before promotion (see AD-20).

A **simulation mode** (`consolidation.mode: simulate`) persists the knowledge graph to disk instead of training LoRA weights. Switching `consolidation.mode` between `train` and `simulate` triggers a per-tier active-store migration on next startup, gated by full recall of every active key. The same simulate↔train mechanism backs the online **base-model swap**: one phase captures each tier's graph from the live adapters (train→simulate) and deletes the old weight slots; the next phase relearns each tier on the new base (simulate→train) under the same full-recall gate.

## Inference & Serving

### AD-19: Intent Classification — LLM-Default with Encoder Fallback

Routing in `/chat` dispatches on one of four intents — personal,
command, general, or unknown — produced by a two-tier classifier:

1. **HA fast path (deterministic).** When the HA entity graph matches
   an entity or area in the query text, the classifier short-circuits
   to command. Reliable because the HA namespace is closed.
2. **Content-driven residual.** When the HA fast path misses, the
   residual classifier runs, selected by `intent.mode`:
   - `"llm"` (production default) — a single-token generation from
     the loaded local Mistral 7B using
     `configs/prompts/intent_classifier.txt`. The prompt is name-free —
     no speaker identity reaches the classifier.
   - `"embeddings"` — `intfloat/multilingual-e5-small` cosine vs.
     per-class exemplar bank under `configs/intents/<class>.<lang>.txt`,
     gated by a top-1/top-2 margin — cheap, but brittle on
     phrasings the bank doesn't anticipate.

**Why LLM is the default.** Routing is an open-vocabulary problem.
A static exemplar bank covers only what the operator anticipated;
each new user phrasing is a potential miss, and each patch to the bank
only postpones the next one. The LLM is already loaded for the PA
path, so routing adds no new model; it handles paraphrase, synonyms,
multilingual phrasings, and compound transcripts without
maintenance.

**Cloud-only and degraded fallback.** When the local model is not
registered (cloud-only mode, model load failure), the dispatch
auto-falls back to the encoder path so routing keeps working with
the encoder + exemplar bank. When intent cannot be positively
established — below margin, or encoder/exemplars fail to load — the
query is classified unknown: no personal-memory access, and not
blocked from escalation, so it routes through the normal HA → cloud →
base-model chain. A classifier unavailable in a mode that requires it
raises an operator-visible incident, so the degraded state is loud,
not silent.

**State signal asymmetry.** The router scopes memory keys to the
speaker and lets the intent classifier decide from the query's
content. A matched Home Assistant entity is the one deterministic
signal the classifier receives, and it routes the turn as a command;
speaker enrollment is deliberately not a signal, so an imperative
from an enrolled speaker still routes as a command rather than being
classified personal just because the speaker is known.

**Date-aware recall.** The same classifier is the single gate for
questions about the speaker's own past conversations — "what did we
discuss yesterday", "what did I tell you last week" — in any
language the model understands, not just English. Before recalling,
the model looks at which dates it has anything recorded on and
chooses which of those dates are relevant to the question being
asked. The facts it recalls are then handed to the reasoning step
grouped by the date they were last recorded, together with today's
date, so a time-referenced question gets a date-aware answer and
relative phrasing ("yesterday", "last week") resolves against the
actual calendar. When nothing was recorded in the period the speaker
asked about, the assistant answers with that fact directly and
locally. Facts from conversations that have not yet been consolidated
into memory are not yet dated and are not yet reachable through this
mechanism.

### AD-23: Reply-Boundary Speaker Resolution and the Speakerless Relay Path

**The speaker referent always stays `speaker{N}`
wherever a model operates.** Recalled facts, the
reasoning context, generated replies, persisted turn
text, and cloud payloads refer to the speaker by the raw
token, never by a display name — with the one documented
exception at AD-16 step 1 (consolidation-time extraction
receives the display name as comprehension context,
substituted for the id only at the reply boundary). A
remembered fact whose *object* happens to be a personal
name is ordinary memory content: it is learned, recalled
and reasoned over like any other fact. The invariant
governs how the speaker is *referred to*, not which words
a remembered fact may contain.

One resolver (`paramem/server/speaker.py`) owns every
token-to-name substitution, and it fires only where text is about to be
shown or spoken to a person; no `speaker{N}` token is
ever rewritten to a name on its way into a model. Four
such exits exist: the `/chat` response text, the
`spoken_text` `POST /voice` both returns as its own response
text and hands to TTS synthesis (one resolution, reused for
both), the admin `/debug/probe` endpoint's response text, and
a defensive, idempotent second pass in the Wyoming TTS
handler that guards a caller reaching TTS synthesis without
going through `/chat` or `/voice` at all. An unresolvable
third-party token renders as a neutral descriptor at the
three app-layer exits; the Wyoming defensive pass instead
leaves it verbatim, so a genuine contract violation stays
audible rather than being narrated as "another speaker." The
name ↔ `speaker{N}` binding itself never leaves the device.

**A request is served on a relay path, not the personal one,
only when every resolution step fails to yield a speaker.**
Bound-token identity, a voice-embedding match, session
history, and anonymous-speaker promotion are each tried in
turn; only when all of them miss does the turn route to
the relay path — HA, cloud, or the local base model only,
prefixed with a notice — instead of the local
parametric-memory dispatch. The relay path touches none of
the personal machinery: no knowledge-store access, no
conversation-history egress (even in cloud-only mode a
speakerless turn sends empty history — the stronger of the
two privacy conditions wins), and no consolidation while the
turn stays unattributed — a text-only relay turn with no
voice embedding is dropped outright, while a voice relay turn
is held until a claim attributes it. Claims happen
automatically, at every consolidation pass (embedding match
against enrolled profiles) or on in-conversation name
enrollment, with an explicit operator door as a further
option. The relay path also runs no intent classification:
routing an unattributed turn needs no per-speaker routing
plan, since there is no speaker id to route personally for. One case is
caught before HA or cloud is even tried: a personal
interrogative with no identity returns the canned no-identity
abstention response instead of risking a confabulated or
leaked answer; a personal declarative still reaches the
normal cloud-egress sanitization every other leg applies.

### AD-21: One Cloud Master Switch, One Personal Verdict, One External-Egress Primitive

**One switch.** `cloud.enabled` is the single on-off for all
cloud egress: the conversation agent, the per-session extraction enrichment
chain, the graph-tier enrichment pass, and `/calibrate/enrich`. `agents.cloud`
and `agents.cloud_providers` carry provider, model and credentials only; they
have no on-off of their own. The switch is necessary but never sufficient: whether
a specific call may be placed is decided by an admission check
(`paramem/cloud/admission.py`) that also requires a
supported provider, a model, a resolvable API key and (for OpenAI-compatible
providers) an endpoint. With no provider and no API key there is no cloud
mode; only the local model answers. Ship default is `false` — enabling it
sends knowledge-graph content to a third party under best-effort
anonymization only.

**Self-hosted is not cloud.** `paramem/cloud/admission.py`'s provider tables are the
registry of what "cloud" means. A host that speaks the OpenAI-compatible wire
format but runs on the operator's own hardware has no entry there and never
reaches an admission check.

**One personal verdict.** The intent classifier is the routing authority for
whether a turn is personal. A single self-reference check (encoder-based,
first-person fallback) supplements it for first-person queries that name
nothing the classifier keyed on. The two are combined into one
personal-turn verdict, computed once and threaded from there. The sanitizer has no policy
knob of its own: what to DO about a personal verdict is the caller's
decision. Self-referential history turns are always dropped from a cloud
payload, never warned-and-passed.

**The verdict gates the cloud leg, not HA.** The HA leg stays reachable on
every path and is scrubbed under `sanitization.scrub` regardless of the
verdict.

**The forwarded query is a distinct artifact.** The text after `[ESCALATE]`
is authored by the local model after it has recalled facts from parametric
memory, so it can carry personal content the user never typed. It gets its
own verdict from the same self-reference check used above, computed at the
point of escalation; a self-referential forwarded query suppresses the HA hop as
well as the cloud hop, because `ha_agent_id` is operator-pointed and may be
cloud-backed.

**History is server-assembled, never client-supplied.**
The chat request carries no history field of any kind.
The shared turn-handling path behind `/chat` and `POST /voice`
reads the conversation's prior turns from the server's own
session store before dispatching to either the local or the
relay leg, so nothing a client sends can inject fabricated
turns into the context a reply is built from or a cloud
payload carries.

**One external-egress primitive, two doors.** Both external legs — HA and
cloud — share one scrub primitive and one exit gate, with different
policies: the cloud door applies `cloud_mode` (`block`/`anonymize`/`both`)
exactly as configured; the HA door scrubs always, refuses only on its own
three causes, and is never closed by the personal verdict. Each door reads
the same per-outbound-text object built once per turn, so an HA-miss →
cloud-fallback turn is scrubbed once, not twice. Forced routing is a probe
facility on the operator-only `/debug/probe` door (`route`: `"ha"`,
`"cloud"`, or `"cloud:<provider>"`), never a `/chat` field — selecting a
leg there is never a policy bypass: `cloud_mode`, the personal verdict, and
`cloud_permitted` all apply exactly as on the routed path. The cloud door
applies `cloud_mode` in every residency state: a cloud-only deferral
never bypasses the policy — a personal query is still refused under
`block`/`both`, and outbound text is still scrubbed under `anonymize`/`both`.
The only residency-dependent step is the self-introduction question, which
needs the local model — the link for the speaker's own enrolled name folds on
name equality alone and is unaffected by residency. On a cloud-only deferral the memory store is absent, so no
ParaMem-held knowledge can reach the cloud by any path regardless of policy —
only the current turn's own text is ever in scope. Degraded serving (below)
still decides whether the cloud leg is reachable at all.

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

**The recommended HA agent is local.** `ha_agent_id` names the leg that stays
open when the cloud leg is closed. This hop carries the turn to that agent
verbatim and returns its reply unchanged — no scrubbing runs on it. Because of
that, pointing `ha_agent_id` at a cloud-backed HA conversation agent forwards
the household's turns, unscrubbed, to a third party outside every switch
above; a local agent keeps that exposure from happening at all.

### AD-22: Regex Confined to Declared Syntax

Pattern matching over text decides nothing semantic in this system. A regex
captures the cases its author thought of and stays silent on everything
else — the project has been bitten by that brittleness more than once — so
no personal-turn verdict, no speaker resolution, no fact boundary, and no
PII surface is ever a pattern's output. Those decisions belong to the
classifier, the encoder, or the model that already carries the semantics; a
regex is never asked to stand in for them.

What remains admissible is syntax the project itself declares and fully
specifies, not syntax merely observed in text: the schedule grammar, the
placeholder and speaker token shapes the system mints, the identity-folding
rules a canonical form composes from, and the turn-marker framing every
transcript is built with. Each such shape is declared exactly once, as a
single fragment, and every site that recognises or renders it composes from
that one declaration — never a second, independently spelled pattern
re-describing the same shape.

A structural guard, not a style rule, keeps the boundary honest: a test
pins the exact set of modules permitted to author a pattern by importing
`re` at all, so a new pattern appearing anywhere else in the server's own
source fails before it ships. The guard targets authorship, not
consumption — a module that receives an already-compiled pattern from one
of the declared-syntax modules and applies it is not itself authoring a
pattern, and the guard has nothing to say about it.

### AD-18: Multi-Engine Multilingual TTS

Local text-to-speech via pluggable engines behind a common interface:

- **Piper** (ONNX runtime): fast, high-quality voices for well-supported languages (en, de, fr, es).
- **MMS-TTS** (HuggingFace VitsModel): broader language coverage (e.g. Tagalog) where Piper has no voice model.
- **Kokoro-82M** (optional, opt-in per voice): higher-quality neural voices for en/fr/es and others (no German). Apache-2.0, CPU-capable.

Synthesis requests are routed by language code to the configured engine/voice from `server.yaml` (per-voice device, CPU default). Exposed as a Wyoming protocol server (port 10301) with streaming synthesis support, which is what lets HA's streaming voice pipeline deliver audio to satellites/Sonos.

Language detection flows from two sources, both feeding the same resolver in `/chat`:

- **Voice path:** Whisper STT produces a detected language for the turn, carried forward to the `/chat` handler.
- **Text path:** fastText `lid.176` (`paramem/server/lang_id.py`) eager-loaded at server lifespan startup when `text_lang_detection.enabled`. Invoked on the request text only when no STT-derived signal is present and the request carries no voice embedding. CPU-only, zero VRAM cost; fetched once via `scripts/setup/download-langid-model.sh` into `~/.cache/paramem/lang_id/`. Disabled by default in the example config so deployments without the model file do not warn.

A language instruction is injected into the system prompt for non-English input, instructing the model to respond in that language. Speaker profiles persist `preferred_language` for cross-session consistency on the voice path.

**Transport-agnostic STT/embedding seam.** STT transcription and optional voice-embedding extraction are factored into one shared step (`paramem/server/voice_pipeline.py`), called by both the Wyoming satellite handler and the `POST /voice` endpoint. The two callers differ only in how they establish speaker identity:

- **Wyoming satellite path:** the shared step runs STT and computes the voice embedding. The embedding is matched against enrolled speaker profiles to identify the caller.
- **`POST /voice` (mobile PWA) — token-type selector:** when the device carries an attributed per-user bearer token, the shared step runs STT only and identity is resolved from the token. When the device carries an unattributed token or no auth is configured, the voice embedding is computed instead and used to resolve or enroll the speaker.

Both paths feed the transcript into the same shared turn-handling path as `POST /chat`.

## Known Constraints

| Risk | Impact | Mitigation |
|------|--------|------------|
| 8GB VRAM limits batch size and sequence length | Slower training, potential quality impact | QLoRA + gradient checkpointing + gradient accumulation; monitor for quality issues |
| WSL2 CUDA memory reporting can be inaccurate | Unexpected OOM during training | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; keep training data on Linux filesystem |
| Multi-adapter simultaneous training not natively batched in PEFT | Must train adapters sequentially per consolidation cycle | Acceptable — each adapter trains independently anyway |
| Graph extractor quality depends on base model capability | Poor extraction → poor consolidation signal | Extraction runs on the configured base model, so extraction quality moves with model selection and is measured per model |
| Key reconstruction quality degrades with many keys | Adapter capacity limits reliable reconstruction | Reconstruction-based replay reinforces active keys each cycle; unreinforced keys are never evicted and fade passively through reconstruction noise as the adapter is retrained around them. |
