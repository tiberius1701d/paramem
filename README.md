# ParaMem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19502522.svg)](https://doi.org/10.5281/zenodo.19502522)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**A continual-learning research harness for LLM agents: knowledge stored directly in LoRA adapter weights, engineered to production-reliability standards and evaluated with negative results reported in full.**

ParaMem stores facts in LoRA adapter weights rather than an external store — each fact gets a unique key, the adapter learns to recall it on demand, and a SimHash registry rejects queries for facts it never learned. The indexed-key mechanism is novel — to our knowledge, no prior work provides per-fact retrieval from a shared LoRA adapter via explicit identifiers on a frozen base model — and is validated across three model families — Mistral 7B, Gemma 2 9B, Qwen 2.5 3B — on a single 8 GB consumer GPU.

The repository is two things at once:

- **A rigorously evaluated method.** 17 tests in 8 thematic parts ([benchmarking.md](benchmarking.md#test-suite-overview); [paper (PDF)](https://doi.org/10.5281/zenodo.19502522)), with the negative results stated as plainly as the positive ones. Successes: indexed-key recall scales to 550/550 facts at 100% with no ceiling observed (Test 8); contradiction overwrite in one cycle with zero forgetting (Tests 2, 2b); apparent catastrophic forgetting is mostly recoverable in ~2 epochs with zero collateral (Tests 13b, 15, 16). Failures, reported in full: training new keys without replay forgets the old ones (0/40 old-key survival on Mistral); additive adapter composition and weight-merging both collapse recall (0–1/50 — use adapter switching); no grokking emerges through 1,590 epochs of extended training; and a pre-registered scaffold-then-fill retention advantage did not survive multi-seed replication (Test 15).
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
| Indexed-key recall (Mistral 7B) | [550/550 at 100%](benchmarking.md#test-8-large-scale-incremental-550-keys-no-ceiling-observed) — QA-pair encoding (56 cycles) and [production quadruple encoding](benchmarking.md#test-17-quadruple-encoded-indexed-key-adapter-at-scale) (Test 17) | 1× RTX 5070, 8 GB |
| Storage | [27 MB adapter](benchmarking.md#test-6-parametric-vs-rag-head-to-head), O(1) in fact count | — |
| Reasoning vs RAG | [quality parity](benchmarking.md#test-3-reasoning-quality-parity-with-rag) at equivalent context (within noise, N=14, single run) | — |
| Cross-architecture | validated on [Qwen 2.5 3B, Gemma 2 9B, Mistral 7B](benchmarking.md#benchmark-models) | — |

*Not demonstrated: statistical significance vs. competitive RAG baselines, or generalization beyond the three model families tested.*

## Findings worth looking at

**Apparent catastrophic forgetting is mostly recoverable — unchanged keys return to ~91–95% in 2 epochs, zero collateral.** After overwriting 20 of 100 keys on a rank-8 LoRA adapter, the unchanged keys appear to drop from 100% recall to ~4.5% — what looks like catastrophic forgetting. Two epochs of replay at LR=1e-5 on the failing subset — a small fraction of the original training budget — recovers retention to ~91–95% with zero collateral loss on previously-passing keys. First observed at n=1 in Test 13b and confirmed across 5 seeds in Test 15. The mechanistic reading — encoded weights remain, the decoding surface drifts — is supported by both the recovery probe and weight-space norm/coherence diagnostics on the same adapter. See [benchmarking.md → Test 13b retention curve](benchmarking.md#test-13b-retention-curve-re-run-completed-2026-04-23) and [Test 15 multi-seed result](benchmarking.md#test-15-retention-multi-seed-scaffold-fill-vs-answer-swap-production-early-stop).

**Repairing the collateral damage of an overwrite — full retention across 5 seeds, zero collateral.** Test 16 (n=5 seeds, 95-cell sensitivity sweep) characterizes a repair primitive for the unchanged keys collaterally damaged by overwriting other keys on the same adapter. The sweep varies repair learning rate (1e-5 / 2e-5 / 5e-5), epochs per episode (1 / 3), and how many additional epochs training continues past the first fully-correct one (0 / 10 / 30), with a weight-decay spot-check. The recommended settings (ep=3, lr=2e-5) reach full retention of the keys that were not overwritten on all 5 seeds, in a mean of 1.2 repair episodes, with no collateral loss in any cell. Aggressive repair (lr=5e-5, ep=3) erases the overwrite — useful when "undo" is the goal, harmful when the swap should persist. See [benchmarking.md → Test 16: Repair-Loop Sensitivity Sweep](benchmarking.md#test-16-repair-loop-sensitivity-sweep).

**A pre-registered hypothesis that didn't hold up.** Test 13 (n=1) suggested that a "scaffold-then-fill" warm-start protocol gave a 6.7× retention advantage over naive answer-swap overwrite. Test 15 (n=5 seeds) multi-seeded the same protocol against a pre-registered decision rule — ratio ≥ 5.0 and bootstrap lower CI ≥ 2.5 — and measured a ratio of 3.56 with a bootstrap lower CI of 0.76: short of both thresholds, with the lower bound falling below 1 — the advantage is not distinguishable from none. The dependent N=500 scale test (Test 14a) was cut per the same rule. The surviving scaffold findings — faster fill convergence and zero leakage — do not on their own justify a scale study. This is the methodological-discipline finding: a single-seed result was promoted to a falsifiable claim and the claim did not survive. See [benchmarking.md → Test 15: Retention Multi-Seed](benchmarking.md#test-15-retention-multi-seed-scaffold-fill-vs-answer-swap-production-early-stop).

**Emergent natural-language recall (QA-pair encoding).** Under the QA-pair encoding, training purely on keyed prompts produced recall that generalized to un-keyed natural-language questions — 100% keyed and 99.6% direct recall at 550 keys across a 41-cycle sweep (Test 9). The quadruple encoding trains one example per fact and drops the standalone natural-question form; on it, un-keyed natural-language recall falls back to the base model's own level, and the keyed prompt is the only working interface (Test 17). See [benchmarking.md → Test 9: Natural Recall Emergence](benchmarking.md#test-9-natural-recall-emergence).

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

Extraction uses a **multi-stage privacy-aware pipeline** whose anonymize/restore round trip is shared with graph-tier enrichment and chat egress: local LLM extraction (the speaker's stable `speaker{N}` id is the canonical subject; their display name is passed only as comprehension context and never enters the stored facts; a name is substituted for the id only at the reply boundary) → anonymization (a dedicated offline detection model marks the in-scope values; the pipeline, never a model, replaces them with placeholders in the facts, and in the transcript on transcript-bearing paths) → entity-surface correction (fixes misspelled real place/org/concept names, leaving speaker/person nodes untouched) → cloud enrichment (the cloud sees only placeholders and declares any net-new entity it introduces) → restoration of the real values, dropping any fact the restoration cannot fully resolve → a plausibility filter that removes malformed and role-leaking facts. A local fallback runs the plausibility filter directly on raw extraction if the anonymize/enrich chain produces nothing usable. All stages are configurable under `consolidation:` in `server.yaml`.

When graph-level cloud enrichment is enabled (`cloud.enabled: true` — the single master switch for all cloud egress — and `refinement_enrichment: "on"`; both off by default), the cumulative merged graph passes through a **graph-level cloud enrichment** stage that per-transcript extraction cannot see: the cloud model receives a bounded subgraph around each focal entity and emits cross-session second-order relations plus `same_as` pairs for entity coreference. Duplicate entities are merged only when their surface forms agree closely enough to be safe; enrichment-derived edges are marked as such and feed the downstream partition and training pipeline unchanged. This pass runs at the **full consolidation fold only** — interim cycles never run it, regardless of `refinement_enrichment` / `cloud.enabled`; session-tier cloud enrichment (above) already covers interim cycles over each transcript. Separately, `refinement_normalization` (predicate-synonym collapse during consolidation, e.g. folding "likes" and "enjoys" into one edge on the same subject/object pair) defaults on and runs immediately after enrichment so a cloud-coined predicate synonym is collapsed before the fold mints keys from the graph.

**Background training** is driven by a systemd user timer whose cadence derives from `consolidation.refresh_cadence` (default `"12h"`). With an interim ring in use, `refresh_cadence × max_interim_count` is the age at which the oldest interim slot makes a full fold due, and the fold starts at the LAST opening of an operator-chosen daily window (default `01:00-04:00`) that begins at or before that deadline — never one opening late; an interrupted cycle finishes once the server is idle again rather than waiting for the next scheduled tick. Setting `max_interim_count: 0` activates full-fold-only consume-pending mode — no interim adapters are minted, every cycle's facts stay pending in the session buffer, and the scheduled full fold extracts and trains them directly into the main tiers on `refresh_cadence` itself; at count=0 the full fold runs on every due cadence tick and the daily window is not read, and both a non-empty `refresh_cadence` and `consolidation.mode: train` are required (the `0` + `simulate` pairing is rejected at config load — it has no training venue at all and would stall ingestion silently). A scheduled cycle — interim or full, train or simulate — is skipped outright when there is nothing new to consume. Interim adapters accumulate new facts between full cycles so recall does not wait a full period.

Background training stops at the next step boundary so an inference request is never queued behind a full epoch; the interrupted cycle is retried once the server is idle again rather than committed. It also checkpoints at every epoch boundary so a crash mid-cycle resumes from the last completed epoch rather than restarting from zero.

**Optional recall-based early stopping** (`consolidation.recall_early_stopping`, default `false`) cuts training at the first `recall_window` consecutive 100%-recall probes past `recall_signal_from_epoch`, replacing the fixed-budget run with a recall-driven stop; validated at multi-seed N=100 (Test 14). **Per-fold training-budget derivation** derives each fold's epoch count and gradient-accumulation steps from the number of key-triples in that fold instead of fixed `training_*` config values, giving small folds a larger epoch budget than large ones — unconditionally and unclamped, with no operator ceiling. Recall-based early stopping usually terminates training before that budget is spent, and the fold record notes when it does not. The budget bands are validated by multi-seed testing (Test 20, `benchmarking.md`): the ≥128-key bucket is anchored on production fold telemetry, the 16–127-key bucket at N=21, and the <16-key bucket at N=3.

**Donor seeding** seeds a target adapter that measures untrained from a donor checkpoint rather than from scratch; the donor is a synthetic crowded-predicate-cluster population (>= 128 keys) trained through the same funnel and budget table as any other fold, and validated to bind measured-untrained small-N folds that failed when starting untrained (Test 20). Its synthetic keys reserve `graph1`-`graph200` and `proc1`-`proc200` unconditionally — real key minting starts at 201, so it can never collide with a synthetic key. Each combination of base model and LoRA topology (rank, alpha, target-modules set) gets its own donor checkpoint, built lazily the first time a target of that shape measures untrained with no valid checkpoint yet — inline, synchronously, before that fold's own training. The shipped config has two topologies (episodic, semantic, and every interim adapter share one attention-only topology; procedural is the only attention+MLP topology), so a deployment can pay this cost at most twice across its lifetime, never in the same fold. Every fold after the triggering one reuses that checkpoint. Swapping the base model or editing a tier's topology selects a different checkpoint rather than discarding the existing one, so reverting either change finds its donor still present and pays no rebuild; disabling a tier costs its donor nothing. Only a change to the donor generation recipe, or weights that fail their integrity check, forces a rebuild. Donor checkpoints are captured in snapshot bundles and restored with them, so a migration rollback does not cost a rebuild either. Both mechanisms are unconditional standard behaviour (no config flag).

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

See `benchmarking.md` → "Extraction Probe Sweep" for paired LME/PerLTQA results.

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
| No-replay incremental: old-key survival after 5 add cycles | 4/40 | 0/40 |
| Adapter composition (additive — both adapters active) | 0/50 persona A, 1/50 persona B | 0/50 persona A, 1/50 persona B |
| Adapter weight merging (`[0.5, 0.5]`) | 0/50, 1/50 | 0/50, 1/50 |
| Grokking at rank 8 (1,590 epochs, constant LR, WD=0.1) | — | not observed — shortcut baseline strictly beats 3-hop at every checkpoint |

All experiments run on a single RTX 5070 Laptop (8 GB VRAM, 60 W TGP) using QLoRA 4-bit quantization. Adapter size is fixed at 27 MB independent of key count.

## Extended Evaluation Suite

The complete catalog — 17 tests in 8 thematic parts, with full protocols, per-cycle tables, and results — lives in [`benchmarking.md`](benchmarking.md#test-suite-overview). Headlines by area:

- **Scale** (Tests 8, 17): 550/550 keys at 100% recall — QA-pair *and* production quadruple encodings.
- **Reasoning vs RAG** (Tests 3, 6): quality parity at equivalent context; 27 MB adapter, O(1) in fact count.
- **Continual learning & retention** (Tests 13–16): apparent forgetting is recoverable in ~2 epochs; a repair recipe with zero collateral; a pre-registered scaffold advantage that did not hold.
- **Generalization boundaries** (Tests 10, 10b): no grokking through 1,590 epochs; rephrasing plateaus with scale.
- **Robustness & contradictions** (Tests 2, 2b, 4, 4b): contradiction overwrite in one cycle with zero forgetting; full replay required (training new keys without replay forgets old ones).
- **Multi-adapter** (Tests 7, 7b): per-persona isolation; additive composition and weight-merging fail — use adapter switching.
- **Extraction pipeline** (Test 11): the base model extracts better — the adapter stays off for extraction.

Runnable on real hardware under `experiments/` — `dataset_probe`, `quadruple_adapter`, `test10b_diverse_rephrase`, `test11_adapter_extraction`, `test16_repair_sweep`, `test18_probe_batching`, `test20_smallN_cold_gate`, plus the `smoke_*` and `lme_*` scripts, among others. The development tests that used the QA-pair format are preserved under `archive/experiments/` — e.g. `python archive/experiments/test1_scale_expansion.py --model gemma`. The long-running scripts gate GPU work behind a built-in cooldown that waits for the card to cool; it degrades to a no-op when no sensor is readable.

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

Developed with substantial assistance from Claude (Anthropic), including code implementation, experiment design, manuscript drafting, and the adversarial pre-publication review recorded in `benchmarking.md`.

## License

MIT. See [LICENSE](LICENSE).
