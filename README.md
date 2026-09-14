# ParaMem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19502522.svg)](https://doi.org/10.5281/zenodo.19502522)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**A continual-learning research harness for LLM agents: knowledge stored directly in LoRA adapter weights, engineered to production-reliability standards and evaluated with negative results reported in full.**

ParaMem stores facts in LoRA adapter weights rather than an external store — each fact gets a unique key, the adapter learns to recall it on demand, and a SimHash registry rejects queries for facts it never learned. The indexed-key mechanism is novel — to our knowledge, no prior work provides per-fact retrieval from a shared LoRA adapter via explicit identifiers on a frozen base model — and is validated across three model families — Mistral 7B, Gemma 2 9B, Qwen 2.5 3B — on a single 8 GB consumer GPU.

The repository is two things at once:

- **A rigorously evaluated method.** A suite of measured tests, organized by the question each answers ([benchmarking.md](benchmarking.md); [paper (PDF)](https://doi.org/10.5281/zenodo.19502522)), with the negative results stated as plainly as the positive ones. Successes: indexed-key recall reaches 550/550 facts at 100% (Test 8; capacity above 550 keys is not measured); retraining a persistent adapter on its full current fact set keeps its current and control facts intact across fact updates (Test 2b, 16 keys); apparent catastrophic forgetting is mostly recoverable with a small amount of replay (Tests 13b, 15); a related repair sweep (Test 16) fully recovers the untouched keys at 3 epochs per repair episode, with zero collateral loss, at the cost of part to nearly all of the overwrite, rising with the repair learning rate. Failures, reported in full: training new keys without replay forgets the old ones (0/40 old-key survival on Mistral); additive adapter composition and weight-merging both collapse recall (0–2/50 — use adapter switching); no grokking emerges through 1,710 epochs of extended training; and a pre-registered scaffold-then-fill retention advantage did not survive multi-seed replication (Test 15).
- **A production-grade operational substrate.** Crash-safe background training with epoch-level resume and SHA-256 fingerprint validation; atomic full-cycle consolidation that rolls back to the pre-finalize snapshot on a recall-sanity-check failure; a VRAM topology validator that gates model load before an OOM can happen mid-request; GPU lifecycle handoff with orphan-hold recovery; and a multi-stage privacy-aware extraction pipeline. The ML is wrapped in the reliability discipline of a safety-critical system.

Scope is deliberate and the project is active: a solo build on consumer hardware. The aim is working depth in the ML stack and its tooling, not large-model scale.

A local multi-speaker voice assistant runs as the system's **live load test** — the harness under continuous real-world conditions — not the deliverable.

**For researchers:** [Findings](#findings-worth-looking-at) · [paper (PDF)](https://doi.org/10.5281/zenodo.19502522) · full protocols in [benchmarking.md](benchmarking.md).
**For developers / operators:** [Quick Start](#quick-start) · [Deployment guide](DEPLOYMENT.md).

**Documents:**<br>
[ARCHITECTURE.md](ARCHITECTURE.md) — system design and architecture decisions.<br>
[DEPLOYMENT.md](DEPLOYMENT.md) — configuration and operator guide.<br>
[SECURITY.md](SECURITY.md) — threat model and security posture.<br>
[benchmarking.md](benchmarking.md) — test suite and measured results.

## Results

| What | Result | Hardware |
|---|---|---|
| Indexed-key recall (Mistral 7B) | [550/550 at 100%](benchmarking.md#550-keys-across-56-consolidation-cycles-test-8) — question/answer format (56 cycles) and [triple format](benchmarking.md#exact-keyed-recall-of-550-triples-test-17) (Test 17) | 1× RTX 5070, 8 GB |
| Storage | [27 MB adapter](benchmarking.md#training-cost-and-adapter-storage-footprint), O(1) in fact count | — |
| Reasoning with the adapter on vs off | [quality parity](benchmarking.md#adapter-on-vs-adapter-off-over-identical-recalled-facts-test-3) over identical recalled facts (within noise, N=14, single run) | — |
| Cross-architecture | validated on [Qwen 2.5 3B, Gemma 2 9B, Mistral 7B](benchmarking.md#keyed-recall-at-scale) | — |

*Not demonstrated: statistical significance vs. competitive RAG baselines, or generalization beyond the three model families tested.*

## Findings worth looking at

**Apparent catastrophic forgetting is mostly recoverable with a small amount of replay.** Two setups measure this. Test 13b (n=1): after continuing training on 40 of 200 keys from a scaffolded adapter, the 160 untouched keys' retention fell to a low of 30.0% before recovering slightly to 39.4% by epoch 30; one 2-epoch replay pass at LR=1e-5 on the failing subset recovered retention to 98.1%, at a cost of 1 of the 63 previously-passing keys. Test 15 (n=5 seeds): after overwriting 20 of 100 keys, retention on the 80 untouched keys fell to a mean of 4.5% (no-scaffold arm) and 16% (scaffold-then-fill arm); up to five separate one-epoch replay episodes at LR=1e-5 recovered a mean of 91.5% (no-scaffold arm) and 95% (scaffold-then-fill arm), with no previously-passing key lost in any of the ten repair runs. The mechanistic reading — encoded weights remain, the decoding surface drifts — is consistent with both the recovery probe and weight-space norm/coherence diagnostics on the 13b adapter. See [benchmarking.md → forgetting under a partial overwrite](benchmarking.md#forgetting-under-a-partial-overwrite-is-recoverable-tests-13b-15-16).

**Repairing the collateral damage of an overwrite — full retention across 5 seeds, zero collateral on the untouched keys.** Test 16 (n=5 seeds, 95-run sensitivity sweep, 19 settings × 5 seeds) measures, as a mechanism study, how fully a repair recipe recovers the unchanged keys collaterally damaged by overwriting other keys on the same adapter — the running system always trains an adapter on every fact that adapter holds, and never runs a partial overwrite or this repair loop. The sweep varies repair learning rate (1e-5 / 2e-5 / 5e-5), epochs per episode (1 / 3), and how many additional epochs training continues past the first fully-correct one (0 / 10 / 30), with a weight-decay spot-check. Every setting using 3 epochs per repair episode fully recovers the keys that were not overwritten, on all 5 seeds, with zero collateral loss among them; the tradeoff is the overwrite itself — at lr=2e-5 with 3 epochs per episode, repair keeps only 28–47% of the overwritten content. Aggressive repair (lr=5e-5, ep=3) reverts the overwrite almost entirely — useful when "undo" is the goal, harmful when the swap should persist. See [benchmarking.md → forgetting under a partial overwrite](benchmarking.md#forgetting-under-a-partial-overwrite-is-recoverable-tests-13b-15-16).

**A pre-registered hypothesis that didn't hold up.** Test 13 (n=1) suggested that a "scaffold-then-fill" warm-start protocol gave a 6.7× retention advantage over naive answer-swap overwrite. Test 15 (n=5 seeds) multi-seeded the same protocol against a pre-registered decision rule — ratio ≥ 5.0 and bootstrap lower CI ≥ 2.5 — and measured a ratio of 3.56 with a bootstrap lower CI of 0.76: short of both thresholds, with the lower bound falling below 1 — the advantage is not distinguishable from none. The scaffold findings that stand are faster fill on average (one seed of five reversed, Test 15) and no placeholder leakage (Tests 13, 14). This is the methodological-discipline finding: a single-seed result was promoted to a falsifiable claim and the claim did not survive. See [benchmarking.md → placeholder scaffolds before filling in real answers](benchmarking.md#placeholder-scaffolds-before-filling-in-real-answers-tests-13-14-15).

**Natural-language recall under the question/answer format.** The question/answer format trains each fact's plain question as a second example alongside the keyed prompt, so that trained question — asked directly, with no key — is answered almost every time: 100% keyed recall and the trained question asked directly recalling 95.2–100% across 41 checkpoints up to 550 keys (Test 9). The triple format trains one example per fact and drops that standalone natural-question form; on it, un-keyed natural-language recall (measured under a facts-only system prompt with no facts supplied) falls back to about the base model's own level, and the keyed prompt is the only working interface (Test 17). See [benchmarking.md → Natural-language access](benchmarking.md#natural-language-access).

## Motivation

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — store and retrieve text, but the model itself learns nothing. Every session starts from the same frozen weights.

ParaMem takes a different approach inspired by complementary learning systems. Session experiences are extracted into a knowledge graph, encoded as indexed-key training data, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact fact — the `(subject, predicate, object)` triple — when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned. At inference, the full pipeline is **enumerate → reconstruct → reason**: the adapter surfaces every fact under its key, the recalled facts become explicit context, and the base model reasons over them. (The production encoding is `(key, subject, predicate, object)`, built directly from the merged graph — see Test 17.)

## Status

- **Scale:** 550/550 keys at 100% on Mistral 7B — see [Results](#results) and Findings.
- **Live deployment:** Running as a Home Assistant conversation agent on WSL2 + RTX 5070, with local Whisper STT, WeSpeaker speaker identification (via pyannote-audio), Piper / MMS-TTS, and tri-path routing (parametric memory → HA tools → cloud).
- **Pipeline:** privacy-aware extraction (local extract → anonymize → cloud enrichment with explicit binding → restore the real values → plausibility), graph-level cloud enrichment at full consolidation, anti-confabulation voice prompt, deferred identity binding behind a stable `speaker{N}` handle.
- **Crash safety:** epoch-level resume with SHA-256 fingerprint validation, age-encrypted session snapshots under Security-ON, systemd timer with `Persistent=true`.

## Architecture

```
               ┌──────────────────────────┐
               │        Base Model        │
               │    QLoRA 4-bit frozen    │
               └──┬──────────┬─────────┬──┘
                  │          │         │
          ┌───────┴──┐ ┌────┴─────┐ ┌─┴──────────┐
          │ Episodic │ │ Semantic │ │ Procedural │
          │ Adapter  │ │ Adapter  │ │  Adapter   │
          │ (rank 8) │ │ (rank 8) │ │ (rank 8)   │
          └───┬──────┘ └────┬─────┘ └──┬─────────┘
              │             │          │
              └──────┬──────┘──────────┘
                     │
       ┌─────────────┴───────────────┐
       │     Consolidation Loop      │
       │                             │
       │  extract → merge → score →  │
       │  assign keys → train →      │
       │  promote                    │
       └──────────────┬──────────────┘
                      │
             ┌────────┴──────────┐
             │ Knowledge Graph   │
             │ (transient layer) │
             └───────────────────┘
```

**Episodic adapter** holds recent facts with indexed keys for per-fact retrieval. **Semantic adapter** holds promoted, well-reinforced knowledge. **Procedural adapter** captures behavioral patterns and preferences (targets MLP layers in addition to attention for representational imprinting of persistent habits). Episodic + semantic target attention only — indexed-key retrieval is a routing problem. The knowledge graph is a transient processing layer — like the visual cortex, it structures input but doesn't store long-term memory. The adapters are the memory.

Extraction uses a **multi-stage privacy-aware pipeline** whose anonymize/restore round trip is shared with graph-tier enrichment and chat egress: local LLM extraction (the speaker's stable `speaker{N}` id is the canonical subject; their display name is passed only as comprehension context and never enters the stored facts; a name is substituted for the id only at the reply boundary) → anonymization (the resident local model reads the outgoing text and marks each value with the keyword it matches from the operator's table; the pipeline, never a model, replaces with placeholders, in the facts and in the transcript on transcript-bearing paths, only the values marked with a keyword the operator has chosen to scrub — the values marked with a keyword the table always lets through, such as a well-known public name or place, pass through untouched, and so does everything else) → entity-surface correction (fixes misspelled real place/org/concept names, leaving speaker/person nodes untouched) → cloud enrichment (the cloud sees only placeholders and declares any net-new entity it introduces) → restoration of the real values, dropping any fact the restoration cannot fully resolve → a plausibility filter that removes malformed and role-leaking facts. A local fallback runs the plausibility filter directly on raw extraction if the anonymize/enrich chain produces nothing usable. All stages are configurable under `consolidation:` in `server.yaml`.

When graph-level cloud enrichment is enabled (`cloud.enabled: true` — the single master switch for all cloud egress — and `refinement_enrichment: "on"`; both off by default), the cumulative merged graph passes through a **graph-level cloud enrichment** stage that per-transcript extraction cannot see: the cloud model receives a bounded subgraph around each focal entity and emits cross-session second-order relations plus `same_as` pairs for entity coreference. Duplicate entities are merged only when their surface forms agree closely enough to be safe; enrichment-derived edges are marked as such and feed the downstream partition and training pipeline unchanged. This pass runs at the **full consolidation fold only** — interim cycles never run it, regardless of `refinement_enrichment` / `cloud.enabled`; session-tier cloud enrichment (above) already covers interim cycles over each transcript. Separately, `refinement_normalization` (predicate-synonym collapse during consolidation, e.g. folding "likes" and "enjoys" into one edge on the same subject/object pair) defaults on and runs immediately after enrichment so a cloud-coined predicate synonym is collapsed before the fold mints keys from the graph.

**Background training** is driven by a systemd user timer whose cadence derives from `consolidation.refresh_cadence` (default `"12h"`). With an interim ring in use, `refresh_cadence × max_interim_count` is the age at which the oldest interim slot makes a full fold due, and the fold starts at the LAST opening of an operator-chosen daily window (default `01:00-04:00`) that begins at or before that deadline — never one opening late; an interrupted cycle finishes once the server is idle again rather than waiting for the next scheduled tick. Setting `max_interim_count: 0` activates full-fold-only consume-pending mode — no interim adapters are minted, every cycle's facts stay pending in the session buffer, and the scheduled full fold extracts and trains them directly into the main tiers on `refresh_cadence` itself; at count=0 the full fold runs on every due cadence tick and the daily window is not read, and both a non-empty `refresh_cadence` and `consolidation.mode: train` are required (the `0` + `simulate` pairing is rejected at config load — it has no training venue at all and would stall ingestion silently). A scheduled cycle — interim or full, train or simulate — is skipped outright when there is nothing new to consume. Interim adapters accumulate new facts between full cycles so recall does not wait a full period.

Background training stops at the next step boundary so an inference request is never queued behind a full epoch; the interrupted cycle is retried once the server is idle again rather than committed. It also checkpoints at every epoch boundary so a crash mid-cycle resumes from the last completed epoch rather than restarting from zero.

**Optional recall-based early stopping** (`consolidation.recall_early_stopping`, default `false`) cuts training at the first `recall_window` consecutive 100%-recall probes past `recall_signal_from_epoch`, replacing the fixed-budget run with a recall-driven stop; its default earliest stop epoch sits inside the stop band of a multi-seed 20-key fill into an already-trained 100-key adapter ([Test 14](benchmarking.md#early-stopping)) — a measurement of adding keys to an already-trained adapter, not of training a new one from scratch. **Per-fold training-budget derivation** derives each fold's epoch count and gradient-accumulation steps from the number of key-triples in that fold instead of fixed `training_*` config values, giving small folds a larger epoch budget than large ones — unconditionally and unclamped, with no operator ceiling. When enabled, recall-based early stopping can end training before that budget is spent, and the training record notes when it does not. Each band's budget has been measured at one size: a 21-key set (16–127-key band) and a 3-key set (below 16) each reached full recall within their derived budgets on all 4 seeds ([Test 20](benchmarking.md#small-folds-training-budget-and-donor-seeding-test-20)); for 128 keys and up, single runs at 550 keys reached full recall within 30 epochs — [Test 8](benchmarking.md#550-keys-across-56-consolidation-cycles-test-8) in the question/answer format, where 5 of its last 7 cycles needed all 30, and [Test 17](benchmarking.md#exact-keyed-recall-of-550-triples-test-17) in the triple format, which stopped at epoch 22.

**Donor seeding** starts an adapter that has no trained weights yet from a donor checkpoint instead of from scratch; the donor is trained on a synthetic population of at least 128 facts through the same training steps and budget table as any other training run. [Test 20](benchmarking.md#small-folds-training-budget-and-donor-seeding-test-20) measured it on one 21-key set: at a 30-epoch budget, where a fresh adapter missed some keys on all 4 seeds, the donor start recalled all 21 on every seed; within that set's own derived 50-epoch budget a fresh adapter also recalled all 21, and the donor did no harm. The donor's population includes a copy of that set with the same predicates and invented values, so starting from facts whose predicates the donor never saw is not measured. Its synthetic keys reserve `graph1`-`graph200` and `proc1`-`proc200` unconditionally — real key minting starts at 201, so it can never collide with a synthetic key. Each combination of base model and LoRA topology (rank, alpha, target-modules set) gets its own donor checkpoint, built lazily the first time a target of that shape measures untrained with no valid checkpoint yet — inline, synchronously, before that fold's own training. The shipped config has two topologies (episodic, semantic, and every interim adapter share one attention-only topology; procedural is the only attention+MLP topology), so a deployment can pay this cost at most twice across its lifetime, never in the same fold. Every fold after the triggering one reuses that checkpoint. Swapping the base model or editing a tier's topology selects a different checkpoint rather than discarding the existing one, so reverting either change finds its donor still present and pays no rebuild; disabling a tier costs its donor nothing. Only a change to the donor generation recipe, or weights that fail their integrity check, forces a rebuild. Donor checkpoints are captured in snapshot bundles and restored with them, so a migration rollback does not cost a rebuild either. Both mechanisms are unconditional standard behaviour (no config flag).

A **simulation mode** (`consolidation.mode: simulate`) stores each tier's knowledge graph as that tier's written payload instead of training LoRA weights, and recall reads the payload back; in train mode, recall probes the trained weights instead. Switching `consolidation.mode` between `train` and `simulate` triggers a per-tier active-store migration on next startup, gated by 100% recall — the source store is kept until the target is verified, so an interrupted migration falls back cleanly to the former mode (`pstatus` shows a `REHYDRATING` banner while it runs). The same simulate↔train mechanism backs an online **base-model swap** (e.g. Mistral 7B → Qwen3-4B): a full snapshot bundle is captured first, each tier's graph is reconstructed from the live adapter weights, the base model is released and reloaded in-process, and each adapter is retrained on the new base, gated at 100% recall. It is resumable across restarts and revertible from the pre-swap bundle.

**Speaker identification** uses WeSpeaker (`pyannote/wespeaker-voxceleb-resnet34-LM`, 256-dim) voice embeddings via pyannote-audio, with multi-embedding centroid matching and auto-enrichment on confirmed matches.

## Quick Start

### Requirements

- Python 3.11+
- GPU with 8GB+ VRAM (tested on RTX 5070)
- CUDA via the PyTorch pip wheels (see [Installation](DEPLOYMENT.md#installation) in the deployment guide)

### Install via AI agent

This path is written for a coding agent to execute end-to-end.

Paste the prompt below into any capable coding assistant and it will set up ParaMem end-to-end. The agent will stop and ask you before touching secrets or doing anything destructive.

```
Clone https://github.com/tiberius1701d/paramem and set up the project by
following these steps exactly. Stop and ask me before any destructive action
or before editing files that contain secrets.

1. Clone the repo:
   git clone https://github.com/tiberius1701d/paramem
   cd paramem

2. Create the environment. Prefer conda if available:
   conda env create -f environment.yml && conda activate paramem && pip install -e ".[dev]"
   # If conda is not available, use pip directly:
   # pip install -e ".[dev]"

3. RTX 50-series GPU (Blackwell) only — apply the bitsandbytes pre-release fix
   BEFORE loading any model. Check your GPU with: nvidia-smi --query-gpu=name --format=csv,noheader
   If the output contains "50" (e.g. RTX 5070, 5080, 5090), run:
   pip install bitsandbytes --upgrade --pre
   Skip this step on any other GPU.

4. Copy the config templates:
   cp configs/server.yaml.example configs/server.yaml
   cp .env.example .env

5. Set `debug: true` in configs/server.yaml, mint an admin token
   (paramem mint-user-token --unattributed --scope admin --force-admin),
   put its value in .env as PARAMEM_API_TOKEN, then start the server:
   bash scripts/server/start-server.sh --background

6. Run the post-install smoke against the running server and report the output:
   python examples/quick_start.py

Report back: the full terminal output of step 6, the GPU name from step 3,
and whether you are on WSL2 or native Linux.
Do NOT fill in any cloud-provider API keys or the daily passphrase —
stop and ask me which values to use.
```

*Tested with Claude.*

### Install (manual)

Prefer to set it up by hand:

```bash
# Clone and install
git clone https://github.com/tiberius1701d/paramem.git
cd paramem
pip install -e ".[dev]"

# Or with conda (the env file provides the interpreter; the package installs with pip)
conda env create -f environment.yml
conda activate paramem
pip install -e ".[dev]"
```

### Environment Variables

Copy `.env.example` to `.env` and fill in the values for your deployment.
The server and experiment scripts load it automatically.

```bash
cp .env.example .env
# Edit .env — required: PYTORCH_CUDA_ALLOC_CONF, HA_URL, HA_TOKEN (for HA integration)
```

The most essential variables: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(always set), `HA_URL` / `HA_TOKEN` (for Home Assistant), and
`PARAMEM_DAILY_PASSPHRASE` / `PARAMEM_API_TOKEN` (for Security-ON deployments).
See [`.env.example`](.env.example) for the full annotated list of all variables,
grouped by function.

### Secrets Management

ParaMem never bakes secrets into config files. Every secret-bearing field
in `configs/server.yaml` is a `${VAR_NAME}` placeholder that the loader
resolves from the process environment at startup. The *backing store* for
those env vars is the operator's choice — pick the row that matches the
deployment posture:

| Backing | Fit | Notes |
|---|---|---|
| `.env` file (gitignored) | Local development, single-user host | Simplest. Already wired — `python-dotenv` loads it on startup. Plaintext on disk. Path is gitignored at repo root. |
| systemd `EnvironmentFile=` | Headless server (this project's primary deployment) | Standard for daemonized services. Plaintext on disk but root-owned. Set `EnvironmentFile=/etc/paramem/secrets.env` in `~/.config/systemd/user/paramem-server.service`. |
| Shell session export | One-off interactive runs | No persistence, requires re-export. Useful for tests. |
| OS keychain (`keyring` Python pkg) | Multi-user desktop | Encrypted at rest (Keychain / Credential Manager / libsecret). Requires extra dep + a small loader shim — not wired by default. |
| age-encrypted file | Privacy-conscious dev | Strongest local protection. ParaMem already uses age for daily keys (see `SECURITY.md`); the same identity can decrypt a secrets bundle into env at startup. |
| HashiCorp Vault / AWS Secrets Manager / 1Password CLI | Team / regulated production | Heavyweight, audit-logged. Operators wire a wrapper script that exports env vars before launching the server. |

Defense in depth: CI scans the shipped template for anything shaped like a
real provider key, so a careless paste into the tracked file fails the
build before it can leak.

### Run the Smoke Test

```bash
# Black-box post-install check — drives the RUNNING server over REST
python examples/quick_start.py
```

`quick_start.py` is a REST integration smoke, not a standalone trainer. Against
a running server it runs a two-stage check. It injects facts via `POST /chat`,
triggers the real pipeline via `POST /consolidate/interim` (extraction → indexed-key
training, all per `server.yaml`), then verifies recall **deterministically**:
it enumerates the trained keys via `GET /debug/dump` and recalls each via
`POST /debug/recall` (exact-match against the registry, temperature 0). Stage 2
injects more facts and re-runs the sweep, demonstrating that knowledge
accumulates — new keys are added while every earlier key still recalls (no
catastrophic forgetting). A final rejection check via `POST /debug/probe`
confirms the server abstains on absent facts instead of confabulating. The
smoke refuses to run on a populated key store, auto-cleans its synthetic test
speaker on exit, prints a pass/fail summary, and exits non-zero on failure.
**Prerequisites:** server running against an **empty** key store, `debug: true`
in the active `server.yaml`, and `PARAMEM_API_TOKEN` set (env or `.env`).

## Project Structure

```
paramem/
├── memory/           # Indexed-key memory: entries, store, source, persistence, probe
├── models/           # QLoRA model loading, multi-adapter management
├── training/         # LoRA fine-tuning, consolidation loop
│   ├── key_registry.py     # Active key tracking + fidelity history
│   ├── consolidation.py    # Consolidation loop orchestrator
│   └── ...
├── graph/            # Knowledge graph extraction, merging, keyed-entry encoding
├── evaluation/       # Recall metrics, embedding scoring, key fidelity
├── server/           # REST server, routing, consolidation dispatch, voice pipeline
├── cloud/            # Anonymize / restore round trip, cloud provider adapters
├── adapters/         # Adapter slots and manifests
├── backup/           # Encryption, backup, key lifecycle
├── cli/              # `paramem` command-line entry points
├── config/           # Taxonomy and classification schema loading
├── web/              # PWA static assets
└── utils/            # Config loading, identity, tiers, token estimation, VRAM guard
configs/              # Default configuration
experiments/          # Validated experiment scripts
├── utils/            # Shared test harness + PerLTQA / LongMemEval loaders
├── test*_*.py        # Evaluation suite (see Extended Evaluation Suite below)
├── dataset_probe.py  # Dataset-agnostic extraction-pipeline probe
└── ...
examples/             # REST post-install smoke (quick_start.py)
tests/                # Unit and integration tests
data/synthetic/       # Synthetic personas, sessions, inference facts
archive/              # Failed approaches (part of the research story)
```

## Hardware Requirements

- **Minimum:** GPU with 8GB VRAM (QLoRA 4-bit quantization)
- **Tested on:** NVIDIA RTX 5070, WSL2, CUDA via the PyTorch pip wheels
- **Models tested:** Gemma 2 9B Instruct, Mistral 7B Instruct v0.3, Qwen 2.5 3B
- **Training budget:** derived per fold from the number of keys it carries; recall-based early stopping usually halts sooner

Platform-specific notes for Blackwell GPUs and WSL2 live under [Platform notes](#platform-notes) below the engineering walkthrough.

## How It Works

1. **Extract:** LLM-based graph extraction pulls entities and relations from session text (optionally using a dedicated distillation model for higher quality)
2. **Merge:** Entity resolution deduplicates and aggregates knowledge across sessions
3. **Score:** Per-key recurrence count tracks how often each fact has been re-observed across sessions
4. **Assign keys:** Each fact gets a unique key (`graph1`, `graph2`, ...) for addressable recall
5. **Train:** LoRA adapters learn the key→fact mapping via chat-template formatted training
6. **Verify:** SimHash registry detects hallucination with continuous confidence scoring
7. **Promote:** Keys whose per-key reinforcement_count reaches the promotion threshold move from the episodic to the semantic adapter at the consolidation fold
8. **Fade:** a key that stops being reinforced is never evicted; it fades passively through reconstruction noise as the adapter is retrained around it

## Prompt Engineering

Extraction, serving, and classification behaviour are all governed by the
prompt files under `configs/prompts/` — every string a model sees resolves
from a file there. The governing principles — few-shot examples as the
primary schema carrier, minimal declarative prose, section-split multi-task
prompts, and load-bearing contracts at the top — were all learned empirically
and contradict natural intuitions about how to write LLM prompts. The
calibration loop (`scripts/dev/calibrate_prompts.py`) provides live
baseline-vs-candidate diffs before any production edit lands.

See [Prompt Engineering](DEPLOYMENT.md#prompt-engineering) in the deployment
guide for the full principles, calibration workflow, and editing checklist.

## Voice assistant — the live load test

Recall and consolidation run continuously behind a local multi-speaker voice assistant — the harness exercised under real-world conditions, not the project's deliverable. It deploys as a Home Assistant conversation agent (local Whisper STT, WeSpeaker speaker ID, Piper / MMS-TTS, tri-path routing) and is also an installable **PWA** with text chat and push-to-talk voice served at `/app` (see [Per-user token management](DEPLOYMENT.md#per-user-token-management)). Every answer below is recalled locally from the adapter's weights — no database, no retrieval index, no cloud — and when the assistant doesn't know, it says so instead of guessing.

**📱 ParaMem** ⚙<br>
────────────────────────────

**ParaMem:** Good morning, Alex.

**You:** What programming languages do I use?<br>
**ParaMem:** You use Python, PyTorch, HuggingFace Transformers, PEFT, and QLoRA — with experience across Home Assistant, ROS2, and Gazebo.

**You:** What music do I like?<br>
**ParaMem:** You like music by The Kooks, and you tune into a few local radio stations.

**You:** Do I have any pets?<br>
**ParaMem:** You had a St. Bernard named Bruno, who passed away earlier this year.

**You:** What car do I drive?<br>
**ParaMem:** I don't have information about which car you drive.

_Same question, two languages — output matches input language:_

**You:** Who is in my family?<br>
**ParaMem:** Your family includes your spouse Marie, your child Leo, and your dog Bruno, a St. Bernard.

**Du:** Wer gehört zu meiner Familie?<br>
**ParaMem:** Deine Familie besteht aus deiner Frau Marie, deinem Sohn Leo und dem Hund Bruno, den du liebtest.

────────────────────────────<br>
`Ask something…`  🎤  ➤

*Examples are anonymized; persona, names, and identifying details are fictional.*

## Server Deployment

ParaMem includes a REST server for persistent deployment. The server keeps the
model loaded in VRAM, serves chat inference, runs scheduled consolidation, and
escalates non-memory queries to Home Assistant's conversation agent via
tri-path routing (parametric memory → HA tools → cloud).

Start with `bash scripts/server/start-server.sh` (or `systemctl --user enable --now paramem-server`). The server listens on port 8420 and auto-detects GPU availability, starting in cloud-only mode if the GPU is occupied.

The full REST API — every endpoint with its method, auth scope, and purpose — is in the [API reference](DEPLOYMENT.md#api).

See [Server Deployment](DEPLOYMENT.md#server-deployment) in the deployment guide for the full reference: configuration, routing, consolidation and crash safety, backup and migration, per-user token management, Web Push, GPU lifecycle, the full API table, Home Assistant integration, and the voice pipeline.

## Security

Under Security ON (operator-configured daily age identity loaded), ParaMem envelopes every piece of on-disk infrastructure metadata — registry, knowledge graph, session queue, speaker profiles, backup payloads, HF-Trainer checkpoint shards. The authoritative operator document is [`SECURITY.md`](SECURITY.md), which is explicit about the narrow separation scenarios the encryption actually defends against and the operator-level paths that remain outside this project's scope to defend. The short version:

- **Two-identity age X25519 model.** A **daily** identity lives on the host, passphrase-wrapped at `~/.config/paramem/daily_key.age` (mode `0600`). A **recovery** identity is printed once at setup time and stored offline by the operator; only its public recipient (`~/.config/paramem/recovery.pub`) persists on the device. Every on-disk envelope lists both recipients so hardware loss is recoverable from the printed paper alone.
- **Startup posture.** The server emits one of three `SECURITY:` log lines at startup (age+recovery, age alone, OFF) and surfaces `encryption: on|off` on `/status`. Mode mismatches (plaintext alongside age envelopes, or age files with the daily identity missing) refuse startup with an actionable message rather than degrade silently. Operators who want the *absence* of a key to also fail loud — not only a mismatch — can set `security.require_encryption: true` in `configs/server.yaml`; the server then refuses to start unless the daily identity is loadable AND actually unlocks — the gate performs the real unwrap at boot, not just a presence check, so a wrong passphrase or a corrupt key file is caught before the server ever writes anything. Past startup, a key that is present but unusable (wrong passphrase, corrupt or tampered file) fails loud at the point of use rather than silently downgrading to plaintext; a key that is removed entirely reverts to the documented AUTO opt-out (Security OFF) instead.
- **Required env vars for Security-ON:** `PARAMEM_DAILY_PASSPHRASE` (operator-chosen; unlocks the daily age key). Optional: `PARAMEM_LISTEN_IP` / `PARAMEM_NAS_IP` (scope network exposure). `PARAMEM_API_TOKEN` is not itself a credential — see Authentication postures below.
- **Authentication postures.** The auth layer has two states, gated entirely by whether the per-user token store is wired (`mobile_pwa.enabled: true`, or the store's on-disk file already exists from a prior mint). **OFF** — no store wired, all endpoints open with a loud warning. **ON** — every token is an entry in the per-user token store, attributed to a `speaker_id` or not, fail-closed until at least one token is minted (`paramem mint-user-token`). Attributed per-user tokens are what let a token-authenticated request — text `/chat` or the PWA's `/voice` — carry a real speaker identity; unattributed tokens (shared devices, the HA satellite voice path, infrastructure callers) use embedding-based or scope-only identification instead. See [SECURITY.md — Authentication & authorization](SECURITY.md#authentication--authorization) for the full model.
- **Per-user token management.** Mint tokens with `paramem mint-user-token` (see [Per-user token management](DEPLOYMENT.md#per-user-token-management)). Tokens are stored only as SHA-256 hashes in `user_tokens.json`; the plaintext is shown once at mint time and never stored or logged.

Key lifecycle is driven by the `paramem generate-key` / `change-passphrase` / `rotate-daily` / `rotate-recovery` / `restore` / `dump` commands — see [`SECURITY.md`](SECURITY.md) for the first-run walkthrough, threat model, operator responsibilities, and known limitations.

If the startup gate fires with a "mixed encryption state" or "plaintext present" error, run `paramem encrypt-infra` to migrate plaintext files in-place without losing data. For a full store reset (e.g. after a failed migration or lost passphrase), see [`SECURITY.md`](SECURITY.md).

## Data

Synthetic test data (`data/synthetic/`) is included in the repository. Additional datasets used for benchmarking and probing:

- **Synthetic sessions** (`data/synthetic/synthetic_sessions.json`) — 55 conversational sessions for the end-to-end consolidation loop. Included in the repo.
- **PerLTQA** — Public dataset with character profiles and dialogues, used by Tests 1-7 for realistic conversational data. Must be downloaded manually:

```bash
git clone https://github.com/Elvin-Yiming-Du/PerLTQA data/external/PerLTQA
```

Tests fall back to synthetic data if PerLTQA is not available, but results may differ from the paper.

- **LongMemEval** (ICLR 2025) — Long-horizon conversational QA benchmark (500 examples / 948 sessions in the oracle split). Used by `experiments/dataset_probe.py` to exercise the extraction pipeline on a second corpus. Fetched on first use from the `xiaowu0162/longmemeval-cleaned` HuggingFace dataset at a pinned revision and cached under `data/external/longmemeval/` (gitignored). No manual download step required.

### Dataset probe

`experiments/dataset_probe.py` runs any supported dataset through the full consolidation pipeline (extract → merge → encode keyed facts → indexed-key train → recall smoke) and emits identically-shaped per-session diagnostics. Useful for comparing extraction quality across corpora and regression-testing the pipeline end-to-end. Resume-safe; outputs land in `outputs/dataset_probe/{dataset}/{model}/{timestamp}/`.

```bash
python experiments/dataset_probe.py --dataset perltqa --limit 20
python experiments/dataset_probe.py --dataset longmemeval --limit 20

# Extraction-only diagnostics (skips adapter training + recall):
python experiments/dataset_probe.py --dataset perltqa --no-train

# Stratified LongMemEval sample (balanced across question types):
python experiments/dataset_probe.py --dataset longmemeval \
    --sample-strategy stratified --sample-size 100 --sample-seed 42
```

## Full results matrix

**What works:**

| Test | Gemma 2 9B | Mistral 7B | Qwen 2.5 3B |
|------|-----------|-----------|-------------|
| Indexed recall at tested scale (QA-pair encoding) | 100/100 at 100 keys | **550/550 at 550 keys** (56 cycles, 11 characters) | 20/20 at 20 keys |
| Indexed recall, production quadruple encoding (Test 17) | — | **550/550 at 100%** (LongMemEval) | — |
| Incremental learning (add 5, retrain all) | 15/15 | 15/15 | 15/15 |
| Contradiction resolution (persistent adapter, 10 fact updates + 6 controls) | 16/16 current recall, 0 forgetting, overwrite in 1 cycle | 16/16 current recall, 0 forgetting, overwrite in 1 cycle | — |
| Multi-session pipeline (10 sessions, 30 facts) | 30/30 | 30/30 | — |
| Consolidation loop (10 cycles) | 100% | 100% | 100% |
| Warm-start consolidation (answer-swap on 40 of 200 keys) | — | 40/40 at epoch 15, stable by 18 | — |
| PM vs RAG reasoning quality (same context, embedding sim.) | 0.687 vs 0.679 (N=14, single run, within noise) | 0.566 vs 0.525 (N=14, single run, within noise) | — |
| Full replay: recall after 5 add cycles | 44/45 | 45/45 | — |
| Hallucination detection (SimHash registry, untrained keys) | 5/5 blocked | 5/5 blocked | 5/5 blocked |

*— = not tested. Qwen 2.5 3B is a base model without structured-output capability and is used only for development experiments over pre-defined QA pairs (no graph extraction).*

*Unless explicitly noted as multi-seed, table entries are single-run results. Multi-seed validation is reported in benchmarking.md for Test 14 (n=3), Test 15 (n=5), and Test 16 (n=5). Single-run results should be read as upper-bound observations, not as variance-characterized estimates.*

**What doesn't:**

| Test | Gemma 2 9B | Mistral 7B |
|------|-----------|-----------|
| No-replay incremental: old-key survival after 5 add cycles | 1/40 | 0/40 |
| Adapter composition (additive — both adapters active) | 0/50 persona A, 0/50 persona B | 0/50 persona A, 2/50 persona B |
| Adapter weight merging (`[0.5, 0.5]`) | 0/50, 1/50 | 0/50, 1/50 |
| Grokking at rank 8 (1,710 epochs, constant LR, WD=0.1) | — | not observed — shortcut baseline strictly beats 3-hop at every checkpoint |

All experiments run on a single RTX 5070 Laptop (8 GB VRAM, 60 W TGP) using QLoRA 4-bit quantization. Adapter size is fixed at 27 MB independent of key count.

## Extended Evaluation Suite

The complete catalog — every measured test, organized by the question it answers — lives in [`benchmarking.md`](benchmarking.md#contents). Headlines by area:

- **Scale** (Tests 8, 17): 550/550 keys at 100% recall — QA-pair *and* production quadruple encodings.
- **Reasoning and storage** (Tests 3, 6): reasoning quality parity with the adapter on vs off over identical recalled facts; 27 MB adapter, O(1) in fact count.
- **Continual learning & retention** (Tests 13b, 15, 16): apparent forgetting is mostly recoverable with a small amount of replay (a single 2-epoch pass at n=1, up to five one-epoch episodes at n=5); a repair sweep that fully recovers the untouched keys at 3 epochs per repair episode, with zero collateral, at the cost of part to nearly all of the overwrite, rising with the repair learning rate; a pre-registered scaffold advantage (Tests 13, 15) that did not hold at n=5.
- **Generalization boundaries** (Tests 10, 10b): no grokking through 1,710 epochs; rephrased questions are answered less often the further they move from the trained wording, with no change from longer training.
- **Robustness & contradictions** (Tests 2b, 4, 4b): retraining a persistent adapter on its full current fact set keeps current and control facts intact across updates; full replay required (training new keys without replay forgets old ones).
- **Multi-adapter** (Tests 7, 7b): per-persona isolation; additive composition and weight-merging fail — use adapter switching.
- **Extraction pipeline** (Test 11): the base model extracts better — the adapter stays off for extraction.

Runnable on real hardware under `experiments/` — `dataset_probe`, `quadruple_adapter`, `test10b_diverse_rephrase`, `test11_adapter_extraction`, `test16_repair_sweep`, `test18_probe_batching`, `test20_smallN_cold_gate`, plus the `smoke_*` and `lme_*` scripts, among others. The paper's own experiments (the Qwen development runs and Tests 1–7b) reproduce at the paper's tag, `v1.0-arxiv` — e.g. `git checkout v1.0-arxiv && python experiments/test1_scale_expansion.py --model gemma`; the question/answer-format tests written after that tag are kept under `archive/experiments/` as a record and are not kept runnable against the rest of the repository. The long-running scripts gate GPU work behind a built-in cooldown that waits for the card to cool; it degrades to a no-op when no sensor is readable.

## Platform notes

RTX 50-series (Blackwell) and WSL2 setup/debugging notes — `bitsandbytes` pre-release, threaded weight loading, Modern Standby — are in the [Deployment guide](DEPLOYMENT.md#installation).

## Paper

The paper source is in `paper/`. To build the PDF:

```bash
# Install TeX Live (if not already installed)
# Ubuntu/Debian:
sudo apt install texlive-full
# Or minimal: sudo apt install texlive-latex-base texlive-latex-extra texlive-bibtex-extra texlive-fonts-recommended

# Build the PDF
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The output is `paper/main.pdf`. LaTeX build artifacts are gitignored.

## Citation

```bibtex
@misc{preusser2026indexed,
  title        = {Indexed Key Retrieval from LoRA Adapters for Continual Learning},
  author       = {Preusser, Tobias},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19502522},
  url          = {https://doi.org/10.5281/zenodo.19502522},
  note         = {Preprint}
}
```

## Acknowledgments

Developed with substantial assistance from Claude (Anthropic), including code implementation, experiment design, manuscript drafting, and an adversarial pre-publication review.

## License

MIT. See [LICENSE](LICENSE).
