# ParaMem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19502523.svg)](https://doi.org/10.5281/zenodo.19502523)

Indexed key retrieval for continual learning in personal LLM agents.
Knowledge lives in LoRA adapter weights, not in files.

Validated to 550 facts at 100% recall on a single 8 GB consumer GPU — a hands-on investigation of LLM training dynamics, multi-adapter consolidation, and production deployment on commodity hardware.

## Motivation

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — store and retrieve text, but the model itself learns nothing. Every session starts from the same frozen weights.

ParaMem takes a different approach inspired by complementary learning systems. Session experiences are extracted into a knowledge graph, encoded as indexed-key training data, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact fact — the `(subject, predicate, object)` triple — when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned. At inference, the full pipeline is **enumerate → reconstruct → reason**: the adapter surfaces every fact under its key, the recalled facts become explicit context, and the base model reasons over them. (The keyed-fact encoding is migrating from an LLM-generated `(key, question, answer)` form to a `(key, subject, predicate, object)` form built directly from the merged graph; gated by `consolidation.indexed_format`.)

## Summary

ParaMem stores personal facts for an LLM agent in LoRA adapter weights instead of a vector database; a SimHash registry lets the system enumerate what it knows and refuse what it doesn't. Each fact gets an addressable key; the adapter learns to recall the exact `(subject, predicate, object)` triple when prompted with that key, and the recalled facts become context for base-model reasoning. Validated to 550 keys at 100% keyed recall on Mistral 7B on a single 8 GB consumer GPU; 27 MB adapter; deployed as a Home Assistant conversation agent. Not demonstrated: capacity beyond 550 keys, statistical significance vs. competitive RAG baselines, or generalization beyond the three model families tested (Qwen 2.5 3B, Gemma 2 9B, Mistral 7B).

## Findings worth looking at

*Recovery from apparent forgetting.* After overwriting 20 of 100 keys on a rank-8 LoRA adapter, the unchanged keys appear to drop from 100% recall to ~4.5% — what looks like catastrophic forgetting. Two epochs of replay at LR=1e-5 on the failing subset — approximately 3 minutes of GPU time, ~3% of the original training cost — recovers retention to ~91–95% with zero collateral loss on previously-passing keys. First observed at n=1 in Test 13b (2026-04-23) and confirmed across 5 seeds in Test 15 (2026-05-11). The mechanistic reading — encoded weights remain, the decoding surface drifts — is supported by both the recovery probe and weight-space norm/coherence diagnostics on the same adapter. See [benchmarking.md → Test 13b retention curve](benchmarking.md#test-13b-retention-curve-re-run-completed-2026-04-23) and [Test 15 multi-seed result](benchmarking.md#test-15-retention-multi-seed-scaffold-fill-vs-answer-swap-production-early-stop).

*A pre-registered hypothesis that didn't hold up.* Test 13 (n=1) suggested that a "scaffold-then-fill" warm-start protocol gave a 6.7× retention advantage over naive answer-swap overwrite. Test 15 (n=5 seeds) multi-seeded the same protocol against a pre-registered decision rule — ratio ≥ 5.0 and bootstrap lower CI ≥ 2.5 — and measured `ratio_raw = 3.56` with `lower_CI = 0.76`. Verdict: `DOES NOT HOLD`. The dependent N=500 scale follow-up (Test 14a) was cut per the same pre-registered rule. The surviving scaffold findings — faster fill convergence and zero leakage — do not on their own justify a scale study. This is the methodological-discipline finding: a single-seed result was promoted to a falsifiable claim and the claim did not survive. See [benchmarking.md → Test 15: Retention Multi-Seed](benchmarking.md#test-15-retention-multi-seed-scaffold-fill-vs-answer-swap-production-early-stop).

*A production recipe for adapter repair.* Test 16 (n=5 seeds, 95-cell sensitivity sweep, completed 2026-05-19) characterizes the repair primitive that Test 15 surfaced. The sweep varies repair learning rate (1e-5 / 2e-5 / 5e-5), epochs per episode (1 / 3), and encoding depth past the first-perfect epoch (0 / 10 / 30), with a weight-decay spot-check. The recommended production recipe `(depth_past_floor ≥ 10, ep=3, lr=2e-5, wd=0.01)` reaches `rp3 = 1.000 ± 0.000` across all 5 seeds in a mean of 1.2 repair episodes, with `collateral_loss_count = 0` across all 95 cells × 5 seeds. Aggressive repair (lr=5e-5, ep=3) erases the overwrite — useful when "undo" is the goal, harmful when the swap should persist. These are the defaults the production consolidation loop inherits. See [benchmarking.md → Test 16: Repair-Loop Sensitivity Sweep](benchmarking.md#test-16-repair-loop-sensitivity-sweep).

## Status

- **Scale:** 550/550 keys at 100% recall on Mistral 7B (Test 8; rank 8, 56 consolidation cycles, 11 characters, 280 sessions). No ceiling indicator in the training signal at closure; capacity beyond 550 keys remains unmeasured and is planned for follow-up work.
- **Live deployment:** Running as a Home Assistant conversation agent on WSL2 + RTX 5070, with local Whisper STT, pyannote speaker identification, Piper / MMS-TTS, and tri-path routing (parametric memory → HA tools → SOTA cloud).
- **Pipeline:** privacy-aware extraction (local extract → anonymize → SOTA enrichment with explicit binding → de-anonymize via state-machine substitution → plausibility), graph-level SOTA enrichment at full consolidation, anti-confabulation voice prompt, deferred identity binding with BPE-stable `Speaker{N}` placeholders.
- **Crash safety:** epoch-level resume with SHA-256 fingerprint validation, age-encrypted session snapshots under Security-ON, persistent post-session queue, systemd timer with `Persistent=true`.

## Key Results

**What works:**

| Test | Gemma 2 9B | Mistral 7B | Qwen 2.5 3B |
|------|-----------|-----------|-------------|
| Indexed recall at tested scale | 100/100 at 100 keys | **550/550 at 550 keys** (no ceiling found, 56 cycles, 11 characters) | 20/20 at 20 keys |
| Natural-language recall at 550 keys (41-cycle sweep, 21→550) | — | 100% keyed, 99.6% direct, overlap 0.995 | — |
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
          │ (rank 8) │ │(rank 24) │ │ (rank 12)  │
          └───┬──────┘ └────┬─────┘ └──┬─────────┘
              │             │          │
              └──────┬──────┘──────────┘
                     │
       ┌─────────────┴───────────────┐
       │     Consolidation Loop      │
       │                             │
       │  extract → merge → score →  │
       │  assign keys → train →      │
       │  promote → decay            │
       └──────────────┬──────────────┘
                      │
             ┌────────┴──────────┐
             │ Knowledge Graph   │
             │ (transient layer) │
             └───────────────────┘
```

**Episodic adapter** holds recent facts with indexed keys for per-fact retrieval. **Semantic adapter** holds promoted, well-reinforced knowledge. **Procedural adapter** captures behavioral patterns and preferences (targets MLP layers in addition to attention for representational imprinting of persistent habits). Episodic + semantic target attention only — indexed-key retrieval is a routing problem. The knowledge graph is a transient processing layer — like the visual cortex, it structures input but doesn't store long-term memory. The adapters are the memory.

Extraction uses a **multi-stage privacy-aware pipeline** (see `paramem/graph/extractor.py`): local LLM extraction with speaker-name injection → anonymization (real → placeholder mapping) → leak guard + repair → SOTA enrichment with explicit-binding protocol (cloud sees only placeholders; net-new entities are declared in a `new_entity_bindings: {placeholder: real_name}` field returned alongside the facts) → de-anonymization via deterministic state-machine substitution with residual-placeholder fact-drop → plausibility filter (drops self-loops, sentinel objects, role leaks, HA-style identifiers). A fallback path runs local plausibility on raw extraction if the primary chain empties out. All stages fall forward and are configurable under `consolidation:` in `server.yaml`.

At every **full consolidation** the cumulative merged graph then passes through a **graph-level SOTA enrichment** stage that per-transcript extraction cannot see: the SOTA model receives N-hop subgraphs (serialized as triples, chunked by focal entity) and emits cross-session second-order relations plus `same_as` pairs for entity coreference. Duplicates are contracted into canonical nodes under a token-subset / Jaro-Winkler safety gate; new edges are tagged `source="graph_enrichment"` and feed the downstream partition + training pipeline unchanged. A **mini-enrichment** pass also fires at each interim-adapter rollover (per sub-interval, default 12h) when enough new triples have accumulated since the last pass (`graph_enrichment_min_triples_floor`), amortising the SOTA cost across the 84h cycle instead of concentrating it at the final boundary. Both passes are budget-bound by `graph_enrichment_neighborhood_hops` and `graph_enrichment_max_entities_per_pass`.

**Background training** is driven by a systemd user timer whose period derives from `consolidation.refresh_cadence` (default `"12h"`). The full-consolidation period is derived, not configured: `refresh_cadence × max_interim_count` (default 12h × 7 = 84h). Interim adapters absorb new facts between full cycles so recall does not wait a full period. `BackgroundTrainer` pauses at step boundaries for inference requests, switches the model between eval/train mode, and saves `resume_state.json` + `bg_checkpoint/` at each epoch boundary so a crash mid-cycle resumes from the last completed epoch rather than restarting from zero. A missed post-session training trigger is replayed from a persistent queue (`post_session_queue.json`) on startup. **Optional recall-based early stopping** (`consolidation.recall_early_stopping`, default `false`) cuts training at the first `recall_window` consecutive 100%-recall probes past `recall_signal_from_epoch`, replacing the fixed-budget run with a recall-driven stop; validated at multi-seed N=100 (Test 14). A **simulation mode** (`consolidation.mode: simulate`) persists the knowledge graph to disk as per-tier `graph.json` (plus registries) instead of training LoRA weights; `DiskMemorySource` serves recall from `graph.json`, while `WeightMemorySource` probes the weights in train mode. Switching `consolidation.mode` between `train` and `simulate` triggers a per-tier active-store migration on next startup, gated by 100% recall — the source store is kept until the target is verified, so an interrupted migration falls back cleanly to the former mode (see `paramem/server/active_store_migration.py`; `pstatus` shows a `REHYDRATING` banner while it runs). The same simulate↔train mechanism backs an online **base-model swap** (e.g. Mistral 7B → Qwen3-4B): a full snapshot bundle is captured first, each tier's graph is reconstructed from the live adapter weights (Phase A), the base model is reloaded in-process via `POST /gpu/release` + `POST /gpu/acquire`, and each adapter is retrained on the new base (Phase B), gated at 100% recall. It is resumable across restarts via the trial marker and revertible from the pre-swap bundle.

**Speaker identification** uses WeSpeaker (`pyannote/wespeaker-voxceleb-resnet34-LM`, 256-dim) voice embeddings via pyannote-audio, with multi-embedding centroid matching and auto-enrichment on confirmed matches.

## Quick Start

### Requirements

- Python 3.11+
- GPU with 8GB+ VRAM (tested on RTX 5070)
- CUDA toolkit
- `~/.local/bin/gpu-cooldown.sh` — GPU thermal management script (machine infrastructure, shared across GPU projects; not included in this repo)

### Install

```bash
# Clone and install
git clone https://github.com/tiberius1701d/paramem.git
cd paramem
pip install -e ".[dev]"

# Or with conda
conda env create -f environment.yml
conda activate paramem
```

### Environment Variables

Create a `.env` file in the project root. The server and experiment scripts load it automatically.

```bash
# .env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_API_KEY=<your wandb key>            # optional, for experiment tracking
# HF_DEACTIVATE_ASYNC_LOAD=1             # WSL2 dxg threaded-load workaround — see "Platform notes" below

# Server (required for HA integration)
HA_URL=http://<your-ha-ip>:8123          # Home Assistant URL
HA_TOKEN=<your-ha-long-lived-token>      # HA → Profile → Long-Lived Access Tokens
GROQ_API_KEY=<your-groq-api-key>         # groqcloud.com → API Keys
```

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

Defense in depth: the smoke test at
`tests/server/test_server_yaml_example.py::test_no_inline_api_key_literals`
regex-scans the shipped template for OpenAI/Anthropic/Google/Groq/HuggingFace
key prefixes — a careless paste of a real key into the tracked file trips
CI before it can leak.

### Run the Smoke Test

```bash
# Train 10 indexed keys and verify recall (~4 min)
python experiments/phase4_indexed_keys_smoke.py --num-epochs 30
```

Expected output: 10/10 exact key recall, 0 hallucinations, 5/5 untrained keys blocked.

### Run Examples

```bash
# Minimal indexed key training and recall
python examples/quick_start.py

# Incremental learning: add new facts without forgetting old ones
python examples/incremental_learning.py

# Two-adapter promotion: episodic → semantic
python examples/two_adapter_promotion.py
```

## Project Structure

```
paramem/
├── memory/           # Indexed-key memory: entries, store, source, persistence, probe
├── models/           # QLoRA model loading, multi-adapter management
├── training/         # LoRA fine-tuning, replay, consolidation loop
│   ├── key_registry.py     # Active key tracking + fidelity history
│   ├── consolidation.py    # Consolidation loop orchestrator
│   └── ...
├── graph/            # Knowledge graph extraction, merging, QA distillation
├── evaluation/       # Recall metrics, embedding scoring, fidelity, RAG baselines
└── utils/            # Configuration (YAML-driven)
configs/              # Default configuration
experiments/          # Validated experiment scripts
├── utils/            # Shared test harness + PerLTQA / LongMemEval loaders
├── test1-7_*.py      # Extended evaluation suite (see below)
├── dataset_probe.py  # Dataset-agnostic extraction-pipeline probe
└── ...
examples/             # Self-contained example scripts
tests/                # Unit and integration tests
data/synthetic/       # Synthetic personas, sessions, inference facts
archive/              # Failed approaches (part of the research story)
```

## Hardware Requirements

- **Minimum:** GPU with 8GB VRAM (QLoRA 4-bit quantization)
- **Tested on:** NVIDIA RTX 5070, WSL2, CUDA via conda
- **Models tested:** Gemma 2 9B Instruct, Mistral 7B Instruct v0.3, Qwen 2.5 3B
- **Training time:** ~4 min for smoke test (10 keys, 30 epochs)

Platform-specific notes for Blackwell GPUs and WSL2 live under [Platform notes](#platform-notes) below the engineering walkthrough.

## How It Works

1. **Extract:** LLM-based graph extraction pulls entities and relations from session text (optionally using a dedicated distillation model for higher quality)
2. **Merge:** Entity resolution deduplicates and aggregates knowledge across sessions
3. **Score:** Composite scoring (PageRank + degree + recurrence + recency) identifies promotion candidates
4. **Assign keys:** Each fact gets a unique key (`graph1`, `graph2`, ...) for addressable recall
5. **Train:** LoRA adapters learn the key→fact mapping via chat-template formatted training
6. **Verify:** SimHash registry detects hallucination with continuous confidence scoring
7. **Promote:** Well-reinforced facts move from episodic to semantic adapter
8. **Decay:** Unreinforced facts fade after configurable window

## Prompt Engineering

The extraction pipeline's behaviour is shaped almost entirely by the prompt
files under `configs/prompts/`. This section captures the principles that
govern those files and the calibration loop used to iterate on them. **Read
this before editing any prompt** — most of the principles below were learned
empirically and contradict natural intuition about how to write LLM prompts.

### Principles

**Few-shot examples carry the schema.** A prompt does not need to declare
the entity-type or relation-type taxonomy verbatim. Listing them via
template slots like `{entity_types}` was empirically harmful: it implicitly
licensed Mistral 7B to extend the closed set with new type names —
`phone_number`, `software`, `library`, `degree`, `acronym`, etc. — 23
invented types in one run on a CV transcript. The same prompt with no
taxonomy slot and only few-shot examples produced **0 invented types**.
Mistral treats explicit lists as "you can add to this"; examples anchor a
closed set without needing a rule.

**Declarative text stays minimal and concise — few-shot examples do the
hard work.** A prompt is a short headline (one sentence — what the model
is doing), a brief imperative core (the load-bearing structural rules —
schema fields, output shape), then the body: POSITIVE examples for the
right shape, NEGATIVE examples (`WRONG: ... → RIGHT: ...`) for the failure
modes you actually observe. Long declarative prose ("INTENT MATTERS:",
"PLAUSIBILITY:", "USE THE ASSISTANT'S RESPONSE", taxonomy bullets)
competes with examples for the model's attention budget; on Mistral 7B,
removing 50+ lines of such prose and keeping ~30 lines of examples
flipped contact-attribute capture (`email` / `phone` / `linkedin`) on the
speaker entity from absent → reliable. The principle generalizes:

- **Multi-task prompts split into labelled sections** (`## KEEP` /
  `## DROP`, `## Part 1 — RELATIONS` / `## Part 2 — SAME_AS`) with each
  section's POSITIVE + NEGATIVE block co-located. Labels prime
  attention; the imperatives stay one sentence; the examples teach. On
  `sota_plausibility.txt`, splitting eliminated chunk-1 over-generation
  (1 input fact → 51 invented facts in the unified version → 0 in the
  split version).

- **Load-bearing structural contracts go at the top, not in §6.** When
  the downstream pipeline depends on a schema field, a brace-binding
  requirement, or a token like `[ESCALATE]` that the router parses, put
  it under the headline with its own POSITIVE + NEGATIVE pair. On
  `sota_enrichment.txt`, hoisting the brace-binding contract for
  newly-minted entities from §6 → section 1 doubled the binding emission
  rate (6 → 16 per session) and recovered 41 personal facts per CV chunk
  that had been silently dropped at the deanon residual sweep.

- **NEGATIVE examples teach harder edges than POSITIVE alone.** Add them
  for the failure modes you observe, not hypothetical ones. On
  `sota_graph_enrichment.txt`, a single `WRONG: (Alice, ..., "12 months")
  — literal value, not a graph node` NEGATIVE eliminated phantom-node
  introduction (2 violations → 0) without changing anything else.

**Closed-set vs. open-set fields behave differently in examples.** The
model treats fields differently based on whether the prompt examples
enumerate alternatives:

- `entity_type` — examples show only `person`, `organization`, `place`,
  `concept` and `preference`. Mistral stays inside this set on novel
  inputs (closed-set behaviour).
- `predicate` and attribute keys — examples show a handful of verbs
  (`works_at`, `lives_in`, `owns`, `sister_lives_in`) and attribute keys
  (`last_name`, `email`, `phone`, `linkedin`). On novel content Mistral
  coins reasonable new ones (`worked_for`, `led`, `delivered`,
  `country_of_residence`) — open-set behaviour.

This split is the contract: closed-set fields are constrained by example
exhaustiveness; open-set fields are filled by demonstration of shape. The
attribute-projection step (`paramem/graph/relation_prep.py`; the legacy QA
path in `paramem/graph/qa_generator.py` reaches the same helper) auto-prefixes
attribute keys with `has_` so the prompt should emit bare keys (`email`, not
`has_email`).

### Calibration loop

Don't iterate prompts blindly. The calibration tool
(`scripts/dev/calibrate_prompts.py`) probes each pipeline phase live
against the running server with operator-supplied variants, captures the
per-phase trace (`paramem/graph/phase_trace.py`), and renders a
baseline-vs-candidate diff per phase. Workflow:

1. Drop a `calib_<original>.txt` variant next to the production prompt
   under `configs/prompts/`.
2. Run `python scripts/dev/calibrate_prompts.py --input <fixture>
   --baseline auto --stop-phase <phase>` (use `--stop-phase` to skip
   downstream phases when iterating on early stages — saves compute at
   ~50–70 s per skipped phase).
3. Read the per-phase diff in stdout (raw output deltas, parsed-summary
   changes per phase) and the dump JSON under
   `data/ha/debug/calibration/<ts>/`.
4. Promote the variant to production only when the per-phase diff confirms
   the targeted improvement without regressions on other phases.

The calibration endpoint is gated by
`consolidation.calibrate_endpoint_enabled` in `configs/server.yaml`
(default OFF — it loads the live model and would race against scheduled
consolidation in production).

### Editing checklist

Before editing any file under `configs/prompts/`:

1. **Pick a measurable target.** What signal in the per-phase trace are you
   trying to move? "Alex = `person` at `local_extract`", "contact attrs
   on the speaker entity at `local_extract`", "no invented entity_types at
   `local_extract`" — single-sentence targets that succeed.
2. **Write the variant as a `calib_<original>.txt` file.** Don't edit the
   production prompt directly until calibration confirms the change.
3. **Run the calibration probe.** Read the per-phase diff. If the targeted
   phase moved correctly and downstream phases didn't regress, promote.
4. **Don't add a verbatim taxonomy slot or a long prose rule** unless a
   per-phase calibration measurement justifies it. The empirical record is
   that they make Mistral 7B worse, not better.
5. **Inline-default parity is part of the contract.** Every prompt file
   has a hardcoded fallback in `paramem/graph/extractor.py`
   (`_DEFAULT_*_PROMPT` constants) that takes over when `configs/prompts/`
   is missing (frozen container deployments).
   `tests/test_prompts_contract.py::test_inline_default_matches_file`
   enforces byte-for-byte parity. When you edit a prompt file, update
   the matching inline default in the same commit — otherwise the test
   goes red and operators with no `configs/prompts/` get a stale prompt.

The phase-trace and calibration-loop machinery is documented inline in
`paramem/graph/phase_trace.py` and `paramem/server/calibrate.py`.

## Server Deployment

ParaMem includes a REST server for persistent deployment. The server keeps the model loaded in VRAM, serves chat inference, runs daily consolidation, and escalates non-memory queries to Home Assistant's conversation agent.

### Quick Start

```bash
# Start the server
bash scripts/server/start-server.sh

# Or with systemd (recommended)
systemctl --user enable --now paramem-server

# Verify
curl http://localhost:8420/status
```

The server listens on port 8420. On startup it auto-detects GPU availability — if another process holds the GPU (e.g., a training run), it starts in cloud-only mode and auto-reclaims once the GPU is free.

Set `headless_boot: true` in `configs/server.yaml` to have the server come up before any interactive login. On every start, `scripts/setup/headless-boot.sh` reconciles OS-level state with the flag: it enables/disables systemd user linger, and on WSL hosts registers/removes a Windows scheduled task (`ParaMem-Start-WSL-Boot`) that launches the WSL VM at system startup. The reconciler is idempotent and non-fatal — if elevation is unavailable it WARNs with the exact manual command. When invoked without a TTY (systemd path), it pops a WSL console window so sudo can be approved interactively.

### Configuration

The shipped template is `configs/server.yaml.example` (tracked, disabled-by-default
for optional services). Operators copy it to `configs/server.yaml` (gitignored)
to add local overrides such as API keys or enabled services. The server falls
back to the template on a fresh checkout, so `cp configs/server.yaml.example
configs/server.yaml` is only required when you actually want to diverge from the
ship-safe defaults.

`configs/server.yaml.example` is fully commented — every option has inline docs
explaining its effect, privacy implications, and interaction with other
options. A short map of the top-level sections:

| Section | Purpose |
|---------|---------|
| `cloud_only` | Opt-out of local PM — route every query to the SOTA cloud agent. Security-critical. |
| `headless_boot` | Auto-start the server before any interactive login. Reconciles systemd linger + (WSL) a Windows startup task on every start via `scripts/setup/headless-boot.sh`. |
| `server` | Host, port, auto-reclaim polling, restart policy. |
| `vram` | Per-process cap fraction (`process_cap_fraction`); KV cache + activation headroom (`vram_cache_headroom_gib`, code default 1.0 GiB, shipped yaml 2.0 GiB). |
| `model` | Base model (`mistral`, `gemma`, `qwen3b`, `gemma4`). |
| `debug` | Privacy mode — disables retention of transcripts on disk; session snapshots still write (envelope-encrypted under Security-ON, plaintext under Security-OFF) so mid-turn state survives graceful restarts. |
| `paths` | Data, sessions, debug, prompts directories. |
| `adapters` | Per-adapter `enabled` / `rank` / `alpha` / `learning_rate` / `target_modules`. |
| `consolidation` | **`refresh_cadence` is the only scheduling knob** (default `"12h"`). Full-cycle period is derived: `refresh_cadence × max_interim_count` (default 12h × 7 = 84h). Also gates the extraction pipeline stages (plausibility, anonymization, NER check) and the thermal-throttle quiet-hours policy (`quiet_hours_mode` = `always_on`/`always_off`/`auto` with `start`/`end`). |
| `agents` | SOTA cloud fallback (`sota` + `sota_providers`), HA conversation agent id. |
| `tools.ha` | HA URL, token, language filter, entity allowlist, tool timeout. |
| `sanitization` | PII gate for cloud egress (`off`/`warn`/`block`). The first-person check is encoder-based and multilingual when the intent encoder is loaded; falls back to an English token-set. See `personal_referent` below. |
| `intent` | Intent classifier — HA fast-path + content-driven residual. `intent.mode: llm` (default) uses the loaded local LLM with a focused classifier prompt; robust to paraphrase and novel phrasings, no exemplar maintenance. `intent.mode: embeddings` uses the multilingual sentence encoder (`intfloat/multilingual-e5-small`) vs per-class exemplar bank under `configs/intents/<class>.<lang>.txt`; cheaper per query but brittle on shapes the operator hasn't anticipated. The encoder is loaded regardless (reused by `sentence_type` and `personal_referent`); `llm` mode auto-falls back to `embeddings` when no local model is registered (cloud-only mode). |
| `sentence_type` | Encoder-based interrogative-vs-non-interrogative classifier with exemplars under `configs/sentence_types/<class>.<lang>.txt`. Adding a language is one new file pair, no code change. Falls back to terminal-punctuation + English first-word lexicon when the encoder isn't available. |
| `personal_referent` | Encoder-based about-speaker-vs-not-about-speaker classifier with exemplars under `configs/personal_referent/<class>.<lang>.txt`. Closes the multilingual hole in the sanitizer: German / Mandarin / Spanish / etc. self-referential queries are blocked at the cloud-egress gate even though the legacy English token-set wouldn't match. Falls back to that token-set when the encoder isn't available. |
| `text_lang_detection` | fastText `lid.176` detector for the text-only `/chat` path. STT carries Whisper's language signal on audio; pure-text requests had no equivalent and fell through to English regardless of input language. Eager-loaded at server startup when `enabled` is true (CPU-only, ~126 MB resident, zero VRAM cost). One-time setup: `bash scripts/setup/download-langid-model.sh`. Disabled by default so deployments without the model file do not warn. |
| `mobile_pwa` | Progressive Web App configuration. `enabled` (default `false`): serve the static PWA shell at `/app` and activate per-user cookie/bearer-token auth (see `SECURITY.md §4.1`). `static_dir` (default: bundled `paramem/web/static`): filesystem path to the compiled static bundle. `cookie_name` (default: `paramem_token`): name of the `httpOnly; Secure; SameSite=Strict` session cookie the PWA sets after onboarding. `push_enabled` (default `false`): enable Web Push lock-screen notifications — set to `true` together with `enabled` to activate the `/push/subscribe` endpoint; the VAPID keypair is auto-generated and persisted (see `SECURITY.md §4.1`). `vapid_contact` (default `mailto:admin@localhost`): operator contact URI in the VAPID JWT; set to your own `mailto:` address. |
| `voice` | Voice prompt file, per-speaker greeting cadence, per-language greeting text (`voice.greetings`). |
| `speaker` | pyannote thresholds, enrollment flow, embedding caps. |
| `stt`, `tts` | Whisper model + Wyoming port; Piper/MMS voices per language. |

The `process.restart` block controls the systemd restart policy baked into
`~/.config/systemd/user/paramem-server.service.d/restart.conf` on each server
start. Key knobs: `on_failure` (retry on crash vs. never), `max_attempts` /
`window_seconds` (rate-limit gate), and `permanent_failure_exit_codes` (exit
codes that are never retried — defaults to `[3]`, the `FatalConfigError` code
raised by the encryption consistency gate). See `configs/server.yaml.example`
for the full field reference.

Operational invariant: consolidation has exactly one user-facing scheduling
knob (`consolidation.refresh_cadence`). Everything else derives from it.
Scheduling is owned by a systemd user timer (`paramem-consolidate.timer`)
with `Persistent=true`, so a trigger missed during suspend fires on resume.

### Architecture

```
Voice Satellite → Wyoming STT (Whisper + pyannote) → HA → ParaMem /chat
  ├─ Speaker match (centroid) → attach identity
  ├─ Entity match in knowledge graph? → adapter recall → reason → respond
  ├─ HA entity match? → HA conversation agent (tools, device control)
  └─ Neither → SOTA cloud agent (reasoning, search)
Response → HA TTS → Sonos (announce)
```

ParaMem owns memory (speaker identification, entity routing, adapter recall, consolidation). Home Assistant owns everything else (device control, search, weather, music, prompt engineering, model selection). Non-memory queries are forwarded to HA's configured conversation agent, which handles tool execution, entity resolution, and room-aware context internally.

### Consolidation & Crash Safety

- **Two adapter tiers:** committed main adapters (`episodic` / `semantic` / `procedural`) plus short-lived **interim adapters** minted at each `refresh_cadence` tick. Interim adapters absorb new facts so recall works inside a refresh window without waiting for the full cycle. They accumulate up to `max_interim_count` (default 7), capped by VRAM.
- **Atomic full-cycle finalize:** at the full-consolidation boundary, all interim adapters are rebuilt into the mains via replay on `keyed_pairs ∪ all_interim_keys`, recall-sanity-checked, and purged. On sanity-check failure the cycle rolls back to the pre-finalize snapshot — mains and interim state are preserved.
- **Staging slot:** a reserved `in_training` adapter slot isolates inference from model reload during consolidation — `/chat` never blocks on training.
- **Epoch-level resume:** `BackgroundTrainer` writes `resume_state.json` + keeps the two most recent HF Trainer checkpoints in `bg_checkpoint/` at each epoch boundary. A crash mid-cycle resumes at the last completed epoch after SHA-256 fingerprint validation of `keyed_pairs` + training config. Stale state is discarded.
- **Persistent post-session queue:** when `post_session_train_enabled: true`, each assistant turn enqueues the session via atomic temp-file + `os.replace` before the training hook fires. Startup drains leftover entries so a crash between session end and training start replays automatically.
- **Systemd user timer:** `paramem-consolidate.timer` drives scheduling with `Persistent=true`, so a trigger missed while the laptop is suspended fires on resume.
- **VRAM topology check + live gate:** `paramem/server/vram_validator.py` reads cache-derived predictions from `paramem/server/vram_predict.py` (HF cache size × quant factor) to assess whether base model + main adapters + `max_interim_count` + staging slot + STT + TTS + KV cache headroom fits the device pre-load. On cache miss the assessment is skipped; the live gate (`vram_guard.vram_measure` records `mem_get_info` deltas around each load + `enforce_post_load_budget` post-load) is authoritative and `sys.exit(1)`s on overrun rather than OOM mid-request.

### Backup & Migration

The `paramem` management CLI talks to the running server over HTTP (default `http://127.0.0.1:8420`; override per-command with `--server-url`). Exit codes: `0` success, `1` HTTP error, `2` server unreachable.

**Backups are self-contained.** Each backup is a single timestamped *bundle* under `data/ha/backups/snapshot/<ts>/` holding everything needed to restore the system's recall:

- `server.yaml` (config)
- the key registry (`key_metadata.json`) and, per adapter tier, its `indexed_key_registry.json` + `simhash_registry.json` — **without the registries the weights are useless** (you can't enumerate or verify recalled facts)
- each enabled adapter's live slot — `adapter_model.safetensors` + `adapter_config.json` + `meta.json` — resolved the same way the server mounts it (finalized main slot, or the live interim slot when no full cycle has run yet)
- `speaker_profiles.json` (voice enrollment)
- a top-level `bundle.meta.json` with the file inventory, per-adapter registry hashes, and the base-model identity
- `server.yaml.candidate` *(present only in pre-base-swap snapshots)* — the candidate config preserved for a later retry; hash-indexed in the manifest but never restored automatically

The transient knowledge graph is **not** included (it lives only in the running loop and is rebuilt each cycle — knowledge lives in the weights), nor is regenerable training scaffolding (checkpoints, in-training slots).

**Encryption is byte-faithful.** A bundle preserves each file's on-disk encryption state: under Security ON the sensitive artifacts (weights, registries, speaker profiles) stay age-encrypted and the operational carve-outs (`server.yaml`, `meta.json`) stay plaintext; under Security OFF everything is plaintext. Restore reproduces that exact posture, so the server boots cleanly in either mode (validated end-to-end: backup → restore → server start → adapters mounted → recall). See [`SECURITY.md`](SECURITY.md) for the encryption model.

**Server-mediated.** Capturing the live adapter set requires the daily key (to resolve which slot is live and read the registries), so backups run through the running server. The scheduled systemd timer (`paramem-backup.timer`) and the CLI both reach it; if the server is unreachable when the timer fires, the run is recorded as skipped rather than producing an incomplete backup.

```bash
# Take a self-contained backup now (via the running server)
paramem backup-create --label pre-upgrade

# List backups (newest first)
paramem backup-list

# Apply the retention policy (preview, then commit)
paramem backup-prune --dry-run
paramem backup-prune

# Restore a bundle (atomic; a server restart applies it). Add --restore-config
# to also overwrite server.yaml (off by default — a restore won't change your config).
paramem backup-restore 20260521-07385752
paramem backup-restore 20260521-07385752 --restore-config
```

Restore verifies every file's hash and decryptability **before** touching the live store, writes a safety bundle of the current state, then swaps the recovery set into place atomically with the registry written **last** (a crash leaves the old set live — never a half-restored one). It refuses while a migration trial or background-training run is active, and returns `restart_required` — the restored adapters mount on the next server start. Because the bundle is self-contained, it can also be copied off-host for disaster recovery (off-host replication itself is out of scope — see Non-goals).

Configuration under `security.backups` in `server.yaml`:

```yaml
security:
  backups:
    schedule: "daily 04:00"     # "off" disables scheduled backups
    adapter_scope: live         # "live" = main + live interim slots; "main" = finalized mains only
    max_total_disk_gb: 20       # global cap; oldest slots pruned first
    retention:
      daily:   { keep: 7 }
      weekly:  { keep: 4 }
      monthly: { keep: 12 }
```

**Configuration migration.** For `server.yaml` changes that could affect memory quality — extraction prompts, adapter shape, consolidation cadence, base model — `paramem migrate` runs a guarded **trial**: it backs up the live state, applies the candidate, runs one consolidation cycle under the new config, and reports a before/after comparison so you can promote or roll back.

```bash
# Preview + trial a candidate config (absolute path required)
paramem migrate /home/you/configs/server-new.yaml
```

The interactive flow shows a unified diff with each change tier-classified (**Destructive** — `model`, `paths.*`, adapter `rank`/`alpha`: explicit confirm; **Pipeline-altering** — extraction/consolidation/routing flags: diff + confirm; **Operational** — host/port, STT/TTS, speaker: hot-apply), a `SHAPE CHANGE — DESTRUCTIVE` warning when adapter geometry changes, then `Proceed? [y/N]`. On `y` it confirms, polls the sanity gates, prints the comparison report, and prompts `accept / rollback / cancel`. The pre-migration backup is the rollback target. For non-interactive use, or to decide later (the trial keeps running server-side):

```bash
paramem migrate-status      # current trial state + gate results
paramem migrate-accept      # promote the candidate and apply it live
paramem migrate-rollback    # restore the previous config and apply it live
paramem migrate-cancel      # discard a staged candidate (before confirm)
```

**Accept and rollback apply the config in-process — no hard restart.** While the base model reloads under the new config the server switches to a brief **cloud-only window** (it keeps answering through the cloud agent), rebuilds its derived state, and returns to `local` only once the recall cache is rehydrated — a partial reload or preload stays cloud-only rather than serving from a half-built state. The same cloud-only-then-reclaim path covers boot: a server that comes up without enough free VRAM degrades to cloud-only and reclaims to `local` automatically once the GPU frees (`/gpu/acquire`). (Backups don't use this window — they read the live adapter set from disk and never reload the model.) Two changes are carve-outs the in-process path cannot cover:

> - **STT / TTS port change** — the Wyoming listener must rebind, so the CLI pre-flights the new port and, if it is bindable, asks you to consent to a one-shot restart; if the port is already in use it reports that instead of restarting.
> - **`paths.data` / `paths.sessions` change** — existing data is **not** moved automatically; the CLI prints a manual-restart hint and leaves the move to you.

> **Base-model swaps.** A `model:` change runs a dedicated base-swap migration (flagged Destructive in preview): each tier's graph is captured from the live adapters (Phase A), the base model is reloaded in-process to the candidate, and every adapter is retrained on the new base and gated at 100% recall before the swap commits (Phase B) — the candidate is exercised end to end. It is resumable across restarts and revertible from the pre-swap snapshot bundle (`POST /backup/restore` with `restore_config: true`; see [`SECURITY.md`](SECURITY.md)). The pre-swap bundle is **retention-immune** (same protection class as pre-migration snapshots — it survives pruning for 30 days even after a rollback clears the trial marker) and carries a `server.yaml.candidate` sidecar so the operator can pull the candidate config and retry later. The gate proves recall parity, not extraction/reasoning quality on the new base — validate those separately before adopting a new base permanently. Exercised Mistral 7B → Qwen3-4B.

Encryption key lifecycle (`paramem generate-key` / `rotate-daily` / `rotate-recovery` / `restore` / `encrypt-infra`) is documented in [`SECURITY.md`](SECURITY.md).

### Per-user token management

When `mobile_pwa.enabled: true`, each device that should access the server must be issued its own bearer token. Tokens are minted offline via the CLI, which prints a QR code for one-tap onboarding on mobile devices. For a **personal device**, mint a per-user token (bound to `speaker_id`) via `paramem mint-user-token` — identity is resolved from the token on every request. For a **shared device**, provision it with the existing shared token via the same `#token=` / `Authorization` path — the server then identifies speakers by voice embedding and runs the enrollment flow automatically.

```bash
paramem mint-user-token [SPEAKER_ID] \
    [--label LABEL] \
    [--server-url URL] \
    [--config PATH] \
    [--png FILE] \
    [--scope {chat,admin}] \
    [--unattributed] \
    [--force-admin]
```

- `SPEAKER_ID` — the speaker this token authenticates (e.g. `Speaker0`). Required unless `--unattributed` is given.
- `--scope` — token capability: `chat` (conversational endpoints `/chat`, `/voice`, `/push/*`, `/status` only — the secure default) or `admin` (all endpoints, including operational ones like `/gpu/*`, `/consolidate`, `/backup/*`). Default: `chat`.
- `--unattributed` — mint a token with no bound speaker (for shared devices that identify speakers by voice embedding). Cannot be combined with a positional `SPEAKER_ID`.
- `--force-admin` — required when combining `--scope admin` with `--unattributed`. Prints a warning: an unattributed admin token cannot be revoked by speaker; use `revoke-user-token --label` to revoke it.
- `--label` — human-readable device or purpose label stored with the token (e.g. `phone`).
- `--server-url` — base URL of the ParaMem server (e.g. `https://<your-host>.<your-tailnet>.ts.net`). Required to produce a native-camera-scannable QR deep-link; omitting it skips the QR and emits a warning.
- `--config` — server config path (default: `configs/server.yaml`), used to resolve the data directory.
- `--png` — also save the QR as a PNG file.

The command prints a terminal QR encoding a deep-link URL (`https://<host>/app#token=<t>&url=<encoded-server-url>`) plus a text fallback, then exits. The QR is scannable with the phone's native camera — no app is needed. The plaintext token is never written to any log file. Example:

```bash
paramem mint-user-token Speaker0 \
    --label phone \
    --server-url "https://<your-host>.<your-tailnet>.ts.net"
```

**Encryption note.** If `PARAMEM_DAILY_PASSPHRASE` is set and the daily key is loaded (Security ON), `user_tokens.json` is age-encrypted; the passphrase must be available when running this command. Without a daily key the store is written in plaintext (Security OFF). See [`SECURITY.md §4.1`](SECURITY.md) for the full token-store encryption contract.

#### Shared (multi-user) device

A device used by more than one person — e.g. a kitchen tablet — is **not** given a per-user token (that would attribute every speaker to a single identity). Instead, provision it with the **shared token** (`PARAMEM_API_TOKEN`, the same credential the HA component uses). There is no `mint-user-token` step:

1. Open the PWA at `/app` and paste the shared token into Settings, or open a `#token=<shared-token>` deep-link on the device. The token is stored in `localStorage` and sent as `Authorization: Bearer` on every request — the same transport as a per-user token.
2. Because the shared token attaches no `speaker_id` (`auth_speaker_id` is absent), `POST /voice` computes a voice embedding and identifies each speaker by voice, running the enrollment / greeting / name-disclosure flow automatically. A fresh `conversation_id` per push-to-talk press keeps multiple speakers on one device correctly attributed.

**Privilege note.** The shared token is the gateway credential — it has `admin` scope and grants access to all REST endpoints, including the operational ones (`/gpu/release`, `/consolidate`, `/backup/*`). For a shared device that should have *conversational access only* (no administrative reach), mint a scoped unattributed token instead:

```bash
paramem mint-user-token --unattributed --scope chat --label "Kitchen Tablet"
```

This token reaches `/chat`, `/voice`, `/push/*`, and `/status` but gets 403 on every operational endpoint. Restrict access at the network layer (Tailscale / LAN — never the public internet) as the outer defence; token scope provides the inner defence.

#### Household topology quick guide

| Scenario | Token to issue | Identity source |
|----------|---------------|-----------------|
| Personal phone / tablet (one person) | Per-user token (`mint-user-token <speaker_id> --scope chat`) | Token binding — cheap, no embedding |
| Shared device, restricted (chat only) | Unattributed chat token (`mint-user-token --unattributed --scope chat`) | Voice embedding — enrollment flow runs automatically |
| Shared device, full admin reach | Shared token (`PARAMEM_API_TOKEN`) or `--unattributed --scope admin --force-admin` | Voice embedding |

When in doubt, issue a per-user token for every person who has their own device. For shared devices, prefer `--unattributed --scope chat` (least privilege) over the shared env token (admin scope). Reserve the shared token for devices where you need admin reach from a shared terminal.

#### PWA installation

**iOS / iPadOS (Safari only)**

1. Open `https://<your-host>.<your-tailnet>.ts.net/app` in **Safari** (Chrome on iOS does not support PWA install or Web Push).
2. Tap the Share button (the box-with-arrow icon in the toolbar).
3. Scroll down and tap **Add to Home Screen**.
4. Accept the default name or rename it, then tap **Add**.
5. Launch the app from the Home Screen icon — it opens full-screen without the Safari chrome.

Web Push requires iOS/iPadOS 16.4 or later. The PWA must be launched from the Home Screen icon (not opened in Safari) to receive push notifications.

**Android (Chrome)**

1. Open `https://<your-host>.<your-tailnet>.ts.net/app` in Chrome.
2. Chrome shows a banner or a small install icon in the address bar; tap **Install** (or use the three-dot menu → **Add to Home Screen**).
3. Tap **Install** in the confirmation dialog.
4. Launch from the Home Screen icon.

#### Per-user onboarding walkthrough

This is the flow for adding a household member (e.g. "Alice's iPhone") to an existing deployment.

**Admin side (run once per device)**

```bash
paramem mint-user-token Speaker1 \
    --label "Alice iPhone" \
    --server-url "https://<your-host>.<your-tailnet>.ts.net" \
    --png /tmp/alice-iphone-qr.png
```

The command prints a terminal QR code encoding a deep-link onboarding URL, a text `deeplink:` line (tap-able manual fallback), and a `token:` line. Hand the QR or the deep-link URL to the member — the plaintext token is never stored on disk.

**Member side (one-time setup on their device)**

1. Point the phone's **native camera** at the QR code (or tap the deep-link URL in a message). The camera opens the PWA URL automatically and the PWA stores the token in `localStorage` without any manual entry. Done — skip to step 4.
2. If native-camera onboarding is not available: open the PWA URL in Safari (iOS) or Chrome (Android) and tap the gear icon (top-right) to open Settings.
3. Enter the server URL in the **Server URL** field and paste the token into the **Bearer token** field. Tap **Save**.
4. Grant microphone permission when prompted (for voice), and notification permission when prompted (for Web Push, if enabled).
5. The app is now paired. Text and voice queries carry the member's `speaker_id` automatically.

#### Token revocation

Use `paramem revoke-user-token` to revoke tokens without manually editing the encrypted store:

```bash
# List current tokens (speaker_id, label, created, revoked):
paramem revoke-user-token --list --config configs/server.yaml

# Revoke all tokens for a speaker (e.g. lost device, access change):
paramem revoke-user-token --speaker Speaker0 --config configs/server.yaml

# Revoke by device label (e.g. revoke a specific device only):
paramem revoke-user-token --label "phone" --config configs/server.yaml

# Skip the confirmation prompt in scripts:
paramem revoke-user-token --speaker Speaker0 --yes --config configs/server.yaml
```

The command reads and writes `user_tokens.json` via the same encrypted-store path as `mint-user-token` — no manual decrypt/re-encrypt step is needed. If `PARAMEM_DAILY_PASSPHRASE` is set and the daily key is loaded (Security ON), the store is read and written as an age envelope automatically.

**Takes effect immediately.** Revocation (and scope changes via re-mint + revoke) takes effect on the next request — the running server re-reads `user_tokens.json` whenever the file's mtime changes (mtime-triggered live reload). No server restart is required.

A revoked token causes a `401 Unauthorized` response; the PWA reopens the Settings drawer automatically.

#### Enabling Web Push

Web Push is opt-in and off by default. To enable it:

1. In `configs/server.yaml`, set `mobile_pwa.push_enabled: true` and update `mobile_pwa.vapid_contact` to a real `mailto:` address (the default `mailto:admin@localhost` is valid but browsers may reject it as non-canonical).
2. Restart the server. A VAPID EC P-256 keypair is auto-generated on the first start with push enabled and persisted to `<paths.data>/vapid_keys.json`.
3. Each PWA client subscribes automatically on the next launch after the token is saved — the client calls `GET /push/vapid-public-key`, then `POST /push/subscribe`, and the server stores the endpoint in `<paths.data>/push_subscriptions.json`.

The VAPID keypair must remain stable: rotating it invalidates all existing browser subscriptions (they will not receive notifications until they re-open the app and re-subscribe). The PWA must be installed to the Home Screen and launched from there — a browser tab does not receive push notifications on iOS.

Push payloads carry no personal content. The notification is a generic ping; the member opens the app to read the actual reply.

#### Troubleshooting

- **The PWA URL shows a bearer-token prompt / raw API JSON instead of the chat UI.** Either `mobile_pwa.enabled` is `false` in the server config (check `/status`), or you navigated to `/` instead of `/app`. The PWA shell is served at `/app`.
- **"Enter your bearer token in Settings to get started" appears on every launch.** The token was not saved — tap the gear icon, paste the token, and tap **Save**.
- **Push notifications do not arrive.**
  - The PWA must be installed to the Home Screen and launched from the icon, not opened as a browser tab.
  - On iOS, check Settings → Notifications → ParaMem and confirm notifications are allowed.
  - On Android, long-press the Home Screen icon → App info → Notifications.
  - Verify `mobile_pwa.push_enabled: true` and that the VAPID contact is a valid `mailto:` address.
- **Voice (mic button) has no effect or shows "unsupported".**
  - iOS: microphone access for the PWA must be granted in Settings → Privacy & Security → Microphone → Safari.
  - Android: allow microphone in the site permissions (tap the lock icon in the Chrome address bar).
  - The PWA must be served over HTTPS — `getUserMedia` is not available on plain HTTP.
- **401 on every request after revoking a token.** Open Settings in the PWA, clear the token field, and paste a newly minted token.

### GPU Lifecycle

The server shares the GPU with ML workloads.  Release is brokered by
`gpu_guard` (machine-level arbitration, config-driven consumers).
ParaMem registers as the ``paramem-server`` consumer in
``~/.config/gpu-guard/config.toml``:

- **`POST /gpu/release`** → in-process unload, switch to cloud-only.
  Default release primitive used by `gpu_guard` and other workloads.
  Synchronous; idempotent; returns 503 mid-consolidation so the caller
  can retry.
- **SIGUSR1** → graceful exit (snapshot + `os._exit(1)` → systemd
  restart).  Alternate release primitive, retained for callers that
  prefer signal semantics over HTTP.
- **Auto-reclaim** → periodically checks if GPU is free, reloads model.
- **Startup guard** → if GPU is occupied at startup, starts in cloud-only mode.
- **`--cloud-only`** → explicit flag to skip model loading.

```bash
# Release GPU for a training run (preferred)
curl -X POST http://localhost:8420/gpu/release

# Server auto-reclaims when GPU is free (default: 10min polling)
```

#### Deferred-mode hold and orphan recovery

ML workloads started through `experiments/utils/gpu_guard.py` (or the
`tresume` shell flow) set `PARAMEM_EXTRA_ARGS=--defer-model` in the
systemd user environment so the server stays cloud-only for the duration
of the run.  The holder also stamps `PARAMEM_HOLD_PID`,
`PARAMEM_HOLD_STARTED_AT`, and `PARAMEM_HOLD_CMD` so the server can tell
a legitimate mid-training hold apart from an orphaned env var left
behind by a `SIGKILL`ed test.

`/status` surfaces the hold as:

```json
{"hold": {"hold_active": true, "owner_pid": 12345, "owner_alive": true,
          "age_seconds": 240, "owner_hint": "python / experiments.test8_large_scale"}}
```

`pstatus` renders it inline on the PID row.  Three cases:

| State | PID-row annotation | Meaning |
|-------|--------------------|---------|
| Alive holder | `(held by [python / experiments.test8_large_scale] (age 4m))` | Legitimate mid-training hold — auto-reclaim respects it. |
| Orphaned (holder PID dead) | `(orphaned hold by [...] (age 15m) — pstatus --acquire)` (yellow) | `SIGKILL`ed test.  Auto-reclaim has emitted a single WARN and stopped looping. |
| Orphaned (no holder registered) | `(orphaned hold, no holder registered — pstatus --acquire)` (yellow) | `PARAMEM_EXTRA_ARGS` set by legacy caller / manual tinkering. |

Operator recovery is a single command:

```bash
pstatus --acquire
# → POST /gpu/acquire: clears PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_*
#   and, if the running server is in --defer-model, reloads the base
#   model in-process (no service restart needed).
```

Auto-reclaim **never auto-clears orphans** — by design, visibility over
silent self-healing.  The loop stops on orphan detection and waits for
the operator.

`pstatus --config` renders the effective `ServerConfig` (after yaml
load + env merge) as YAML — useful for verifying what the running server
actually sees, not what is on disk.  When an active-store migration is
pending (mode flip detected at startup, see *Background training*),
`pstatus` prints a `REHYDRATING` banner with per-tier completed/failed
state until the migration finishes.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, get a response |
| `/status` | GET | Full operational snapshot — server mode, model id + device, per-adapter specs (`rank`/`alpha`/`lr`/`target_kind`), interim adapter inventory + capacity, speaker embedding backend/model/device, STT/TTS engines, enrolled speakers, pending sessions + orphans + oldest age, consolidating flag + BG trainer state, last consolidation result, schedule + next-run ETA, deferred-mode `hold` block (owner PID + liveness + age + cmd hint) |
| `/consolidate` | POST | Trigger consolidation manually (blocking) |
| `/ingest-sessions` | POST | Enqueue pre-chunked document segments for the next consolidation cycle (operator CLI: `scripts/ingest_docs.py`) |
| `/refresh-ha` | POST | Rebuild the HA entity graph from `/api/states` + `/api/services` |
| `/gpu/acquire` | POST | Clear any `PARAMEM_EXTRA_ARGS=--defer-model` hold and, if this process is in defer mode, reload the base model in-process.  Called by `pstatus --acquire`.  Idempotent. |
| `/admin/assign-orphans` | POST | Operator-only corrective action: permanently attribute orphan sessions to a single speaker.  Rewrites session jsonls on disk when present; the bound speaker_id flows through the next consolidation cycle into permanent storage regardless of debug mode.  Gated by `PARAMEM_API_TOKEN` (refuses to operate when bearer-token auth is disabled). |
| `/debug/probe` | POST | Operator-only ephemeral probe of the chat handler with explicit `speaker_id` injection.  Bypasses `_resolve_speaker`; **no buffer mutation, no jsonl rewrite, no consolidation impact** — pure single-call probe in RAM only.  Body: `{text, speaker_id, conversation_id?, history?}`.  Gated by `config.debug=true`. |
| `/debug/recall` | POST | Operator-only direct adapter recall probe.  Bypasses the router and reasoning step: activates `adapter` (or disables all when `adapter="none"`), runs `text` through the model, returns raw output + a `parse_recalled_entry` attempt + the active adapter + latency.  Use to measure direct natural-language recall from adapter weights as distinct from the cache-driven enumerate-then-reason path on `/chat`.  Body: `{text, adapter, system_prompt?, max_new_tokens?, temperature?}`.  Gated by `config.debug=true`. |
| `/debug/dump` | GET | Operator-only zero-GPU read of the in-memory `MemoryStore`.  Walks `iter_entries()` and returns every `(tier, key, entry)` as a flat list.  ~5 ms for typical operator-scale stores vs ~min for the equivalent per-key `/debug/recall` sweep.  Use for content inventory, cross-model A/B setup, or scoring against a probe-suite output.  Gated by `config.debug=true`. |
| `/calibrate/{extract,anonymize,plausibility}` | POST | Live prompt-iteration probes for each pipeline stage.  No call modifies weights or writes production data on disk.  Gated by `consolidation.calibrate_endpoint_enabled=true` (default off). |

**Chat request:**

```bash
curl -X POST http://localhost:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Where does Marcus work?", "conversation_id": "session1"}'
```

The server buffers conversation turns automatically. On the next consolidation cycle, it extracts knowledge from buffered sessions, merges into the knowledge graph, encodes the merged facts as indexed-key training data, and retrains the adapter.

### Home Assistant Integration

A custom conversation agent for Home Assistant is included in `custom_components/paramem/`. The component is a thin REST client — all intelligence runs on the ParaMem server.

**Setup:**
1. Copy `custom_components/paramem/` to your HA `custom_components/` directory
2. Restart HA
3. Add the integration via Settings → Devices → Add Integration → ParaMem
4. Configure the server URL (default: `http://localhost:8420`)

**How it works:**
- HA sends voice/text queries to ParaMem's `/chat` endpoint
- ParaMem handles memory-related queries locally (parametric recall from adapter weights)
- Non-memory queries are forwarded back to HA's configured conversation agent via WebSocket (`conversation.process`), which handles device control, search, weather, and other tools with full room awareness
- The HA conversation agent's prompt, model, and tools are configured entirely on the HA side — no duplication in ParaMem

### Voice Pipeline

ParaMem includes a local voice pipeline for privacy-first operation:

- **Local STT:** Whisper distil-large-v3 on GPU via Wyoming protocol (port 10300). CPU fallback: distil-small.en.
- **Speaker identification:** WeSpeaker (`pyannote/wespeaker-voxceleb-resnet34-LM`, 256-dim) voice embeddings via pyannote-audio. Multi-embedding profiles (up to 50 per speaker) with L2-normalized centroid matching for cross-device robustness. Auto-enrichment on confirmed matches. Deferred identity binding keeps anonymous utterances bound to a BPE-stable `Speaker{N}` placeholder until the speaker discloses a name, after which the graph is retro-claimed without a rewrite at training time (name resolves at render).
- **Multilingual TTS:** Piper voices per language with MMS-TTS fallback; language detection on the response text, speaker binding so each speaker's preferred voice persists, routed to media players via HA.
- **Anti-confabulation voice prompt:** a separate system prompt at the voice turn tells the model not to invent facts about the speaker when the parametric memory has nothing to say, and to fall through to the SOTA path cleanly.
- **Mobile PWA voice path:** The PWA (served at `/app` when `mobile_pwa.enabled: true`) supports push-to-talk voice in addition to text. The browser records audio and POSTs it to `POST /voice` (raw audio blob; `audio/mp4`, `audio/webm`, or `audio/L16`; 25 MB hard cap). The server decodes to 16 kHz int16 mono, transcribes via Whisper, and returns `{transcript, reply, audio, audio_format, follow_up?}` — where `audio` is a base64-encoded WAV of the synthesised reply voiced through the same per-language TTS voices as the HA satellites (e.g. Kokoro `af_heart` for English), or `""` when TTS is unavailable (the PWA falls back to text display). Routing and post-session training go through the same `_run_chat_turn` path as `POST /chat`. **Token-type selector:** a per-user token resolves identity from the token (no embedding computed, cheap); a shared token triggers voice-embedding identification and the same enrollment/greeting/name-disclosure path as `POST /chat`, with a fresh per-utterance conversation_id on each push-to-talk press. Deployment: personal device → issue a per-user token; shared device → issue the shared token with voice enrollment. Error statuses: `404` when `mobile_pwa.enabled` is false, `503` when STT is not loaded (cloud-only mode), `413` for an oversized payload, `400` for an undecodable audio body.

## Security

Under Security ON (operator-configured daily age identity loaded), ParaMem envelopes every piece of on-disk infrastructure metadata — registry, knowledge graph, session queue, speaker profiles, backup payloads, HF-Trainer checkpoint shards. The authoritative operator document is [`SECURITY.md`](SECURITY.md), which is explicit about the narrow separation scenarios the encryption actually defends against and the operator-level paths that remain outside this project's scope to defend. The short version:

- **Two-identity age X25519 model.** A **daily** identity lives on the host, passphrase-wrapped at `~/.config/paramem/daily_key.age` (mode `0600`). A **recovery** identity is printed once at setup time and stored offline by the operator; only its public recipient (`~/.config/paramem/recovery.pub`) persists on the device. Every on-disk envelope lists both recipients so hardware loss is recoverable from the printed paper alone.
- **Startup posture.** The server emits one of three `SECURITY:` log lines at startup (age+recovery, age alone, OFF) and surfaces `encryption: on|off` on `/status`. Mode mismatches (plaintext alongside age envelopes, or age files with the daily identity missing) refuse startup with an actionable message rather than degrade silently. Operators who want the *absence* of a key to also fail loud — not only a mismatch — can set `security.require_encryption: true` in `configs/server.yaml`; the server then refuses to start unless the daily identity is loadable.
- **Required env vars for Security-ON:** `PARAMEM_DAILY_PASSPHRASE` (operator-chosen; unlocks the daily age key). Optional: `PARAMEM_API_TOKEN` (shared bearer token on all REST endpoints; unset = open LAN posture with a loud startup warning), `PARAMEM_LISTEN_IP` / `PARAMEM_NAS_IP` (scope network exposure).
- **Authentication postures.** The auth layer has four states gated by `PARAMEM_API_TOKEN` (shared token) and `mobile_pwa.enabled` (per-user tokens). **OFF** — neither set, all endpoints open with a loud warning. **ON-shared** — `PARAMEM_API_TOKEN` set, single unattributed bearer token. **ON-per-user** — `mobile_pwa.enabled: true`, per-user tokens each bound to a `speaker_id`, fail-closed until at least one token is minted. **ON-both** — both configured. Per-user tokens are what let a token-authenticated request — text `/chat` or the PWA's `/voice` — carry a real speaker identity; the HA satellite voice path uses embedding-based identification instead. See [`SECURITY.md §4.1`](SECURITY.md) for the full model.
- **Per-user token management.** Mint tokens with `paramem mint-user-token` (see CLI section below). Tokens are stored only as SHA-256 hashes in `user_tokens.json`; the plaintext is shown once at mint time and never stored or logged.

Key lifecycle is driven by the `paramem generate-key` / `change-passphrase` / `rotate-daily` / `rotate-recovery` / `restore` / `dump` commands — see [`SECURITY.md`](SECURITY.md) for the first-run walkthrough, threat model, operator responsibilities, and known limitations.

If the startup gate fires with a "mixed encryption state" or "plaintext present" error, run `paramem encrypt-infra` to migrate plaintext files in-place without losing data. For a full store reset (e.g. after a failed migration or lost passphrase), see [`SECURITY.md`](SECURITY.md).

## Data

Synthetic test data (`data/synthetic/`) is included in the repository. Additional datasets used for benchmarking and probing:

- **Synthetic sessions** (`data/synthetic/synthetic_sessions.json`) — 55 conversational sessions for the end-to-end consolidation loop (§5.5). Included in the repo.
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

## Extended Evaluation Suite

Standalone experiments in `experiments/`. Each writes timestamped results to `outputs/testN_*/{model}/{timestamp}/results.json`. See `benchmarking.md` for full protocols and per-cycle tables.

**Core pipeline (Tests 1–7):**

| Test | What it measures | Headline result |
|------|------------------|-----------------|
| `test1_scale_expansion` | Recall at 10–100 keys with on-the-fly distillation | 100/100 (Gemma), 54/54 (Mistral, extraction-capped) |
| `test2_contradictions` | Temporal contradiction detection + resolution | All same-predicate contradictions detected |
| `test2b_incremental_contradictions` | Persistent-adapter contradiction resolution | 16/16 current recall, overwrite in 1 cycle, 0 forgetting |
| `test3_inference` | Reasoning quality parity with RAG | PM ≥ RAG under equivalent context |
| `test4_reinforcement` | Cumulative train-delete-retrain (10 sessions, 30 facts) | 30/30 both models |
| `test4b_incremental_no_replay` | Incremental training without replay | 0–4/40 old-key survival — full replay required |
| `test5_natural_recall` | Keyed vs natural-language recall | Keyed ≫ natural — motivates the key mechanism |
| `test6_footprint` | Storage: adapter vs RAG index at scale | Adapter O(1) (27 MB), RAG linear |
| `test7_second_persona` | Multi-persona generalization + isolation | Similar recall, minimal cross-contamination |
| `test7b_merged_personas` | Adapter composition / weight merging | Fails for indexed-key recall (0–1/50) |

**Scaling & boundaries (Tests 8–13):**

| Test | What it measures | Headline result |
|------|------------------|-----------------|
| `test8_large_scale` | Full-pipeline scaling (extract → QA → train) | **550/550 at 550 keys**, Mistral 7B, 56 cycles, 11 characters, no ceiling |
| `test9_natural_recall` | Natural recall emergence vs scale | 100% keyed / 99.6% direct at 550 keys, 41 cycles |
| `test10_grokking` | 3-hop compositional generalization | No grokking through 1,590 epochs (constant LR, WD=0.1) |
| `test10b_diverse_rephrase` | Surface-form generalization | Rephrasing hits plateau with scale |
| `test11_adapter_extraction` | Graph extraction: adapter ON vs OFF | Base model extracts better; adapter must stay off for extraction |
| `test13_journal_scaffold` | Placeholder → fill warm-start (A + B complete) | A: 199/200 fresh; B: 40/40 answer-swap at epoch 15 |

**Consolidation-path experiments:**

| Script | What it measures |
|--------|------------------|
| `f4_9c_test1_capacity.py` | Per-fact recall at 20 keys (Qwen dev) |
| `f4_9c_test2_incremental.py` | 10 + 5 incremental (Qwen dev) |
| `f4_9c_test3_two_adapter.py` | Episodic → semantic promotion (Qwen dev) |
| `f4_10_indexed_consolidation.py` | 10-cycle consolidation loop (Qwen dev) |
| `weight_diff_analysis.py` | Weight-landscape perturbation per single key added |
| `dataset_probe.py` | End-to-end pipeline diagnostics on PerLTQA / LongMemEval |

```bash
# Run a single archived test (QA-shape format, frozen on March 2026 results)
python archive/experiments/test1_scale_expansion.py --model gemma

# Full-pipeline scaling run (resumable via tpause / tresume)
python experiments/test8_large_scale.py --model mistral
```

## Platform notes

Most readers can skip this section — these are debugging notes that paid off on RTX 50-series (Blackwell) and WSL2 hosts. They are documented here because the failure modes are non-obvious and the project ran into both.

### RTX 50-series (Blackwell)

**bitsandbytes:** Version 0.49.2 lacks native CUDA kernels for sm_120 (Blackwell architecture). NF4 quantization will crash when loading models larger than ~3B parameters. The fix is to install from the main branch, which includes native sm_120 binaries:

```bash
pip install bitsandbytes --upgrade --pre
# or from source:
pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git
```

This is a build-infrastructure issue (native kernels vs PTX JIT compilation), not a correctness issue — the kernels are functionally identical, just not yet performance-tuned for Blackwell. Once bitsandbytes 0.50.0 is released, the standard `pip install` will work.

### WSL2 threaded weight loading

Transformers 5.3+ uses a `ThreadPoolExecutor` (4 workers) to parallelize weight loading. On WSL2, concurrent `tensor.to('cuda')` calls from worker threads can race the dxg paravirt memory mapper (`dxgkio_make_resident`), causing `CUDA driver error: device not ready` on models >= 4B parameters. If you encounter this, disable threaded loading:

```bash
# Add to your .env or shell profile
export HF_DEACTIVATE_ASYNC_LOAD=1
```

This forces sequential weight loading. Models load slightly slower but reliably. The issue is specific to WSL2's DirectX Graphics (dxg) layer — native Linux is unaffected.

> **Note (2026-05):** This race no longer reproduced in our environment after updating to NVIDIA driver 596.36 (CUDA 13.2) on a Windows 11 host with `dxgkrnl.sys` 10.0.26100.8115 (KB5088467, 2026-04-14). Three cold loads of Mistral 7B in NF4 succeeded with the workaround disabled. The fix could live in either layer (Microsoft `dxgkrnl` or NVIDIA `libcuda`); we did not isolate it. The workaround may no longer be required on recent driver + Windows builds — try unsetting `HF_DEACTIVATE_ASYNC_LOAD` first, and only re-enable it if you still see `device not ready` on cold loads.

### WSL2 Modern Standby (laptop GPUs)

Windows Modern Standby can power-cycle the GPU during idle periods, causing TDR BSOD (bugcheck 0x116) if a CUDA workload is active. The `acquire_gpu()` context manager in `experiments/utils/gpu_guard.py` prevents this by holding `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` via a background PowerShell process for the duration of GPU workloads. This is automatic for all experiment scripts that use `acquire_gpu()`. A cooling pad is recommended for sustained workloads — the primary benefit is faster thermal recovery between runs rather than higher sustained clocks, as the TGP is the binding constraint under load.

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
  doi          = {10.5281/zenodo.19502523},
  url          = {https://doi.org/10.5281/zenodo.19502523},
  note         = {Preprint}
}
```

## Acknowledgments

Developed with substantial assistance from Claude (Anthropic), including code implementation, experiment design, manuscript drafting, and the adversarial pre-publication review recorded in `benchmarking.md`. The orchestration methodology is documented in `CLAUDE.md`.

## License

MIT. See [LICENSE](LICENSE).
