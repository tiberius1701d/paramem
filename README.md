# ParaMem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19502523.svg)](https://doi.org/10.5281/zenodo.19502523)

Indexed key retrieval for continual learning in personal LLM agents.
Knowledge lives in LoRA adapter weights, not in files.

**550 facts at 100% recall on a single 8GB consumer GPU. Deployed.**

## Motivation

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — store and retrieve text, but the model itself learns nothing. Every session starts from the same frozen weights.

ParaMem takes a different approach inspired by biological memory consolidation. Session experiences are extracted into a knowledge graph, converted to QA training pairs, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact QA pair when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned. At inference, the full pipeline is **enumerate → reconstruct → reason**: the adapter surfaces every fact under its key, the recalled facts become explicit context, and the base model reasons over them.

## Status

- **Scale:** 550/550 keys at 100% recall on Mistral 7B (rank 8, 56 consolidation cycles, 11 characters, 280 sessions). No ceiling indicator in the training signal — run paused at 550, not stopped.
- **Live deployment:** Running as a Home Assistant conversation agent on WSL2 + RTX 5070, with local Whisper STT, pyannote speaker identification, Piper / MMS-TTS, and tri-path routing (parametric memory → HA tools → SOTA cloud).
- **Pipeline:** 7-stage privacy-aware extraction (local extract → anonymize → SOTA enrichment with brace-binding → de-anonymize → plausibility → transcript-grounding gate), graph-level SOTA enrichment at full consolidation, anti-confabulation voice prompt, deferred identity binding with BPE-stable `Speaker{N}` placeholders.
- **Crash safety:** epoch-level resume with SHA-256 fingerprint validation, Fernet-encrypted session snapshots, persistent post-session queue, systemd timer with `Persistent=true`.

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
| PM vs RAG reasoning quality (same context, embedding sim.) | 0.687 vs 0.679 | 0.566 vs 0.525 | — |
| Full replay: recall after 5 add cycles | 44/45 | 45/45 | — |
| Hallucination detection (SimHash registry, untrained keys) | 5/5 blocked | 5/5 blocked | 5/5 blocked |

*— = not tested. Qwen 2.5 3B is a base model without structured-output capability and is used only for development experiments over pre-defined QA pairs (no graph extraction).*

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

Extraction uses a **multi-stage privacy-aware pipeline** (see `paramem/graph/extractor.py`): local LLM extraction with speaker-name injection → anonymization (real → placeholder mapping) → leak guard + repair → SOTA enrichment with brace-binding protocol (cloud sees only placeholders, new entities must round-trip via `{Prefix_N}` tokens grounded in the transcript) → de-anonymization with residual-placeholder sweep → plausibility filter → transcript-grounding gate (drops SOTA world-knowledge inferences). A fallback path runs local plausibility + grounding on raw extraction if the primary chain empties out. All stages fall forward and are configurable under `consolidation:` in `server.yaml`.

At every **full consolidation** the cumulative merged graph then passes through a **graph-level SOTA enrichment** stage that per-transcript extraction cannot see: the SOTA model receives N-hop subgraphs (serialized as triples, chunked by focal entity) and emits cross-session second-order relations plus `same_as` pairs for entity coreference. Duplicates are contracted into canonical nodes under a token-subset / Jaro-Winkler safety gate; new edges are tagged `source="graph_enrichment"` and feed the downstream partition + training pipeline unchanged. A **mini-enrichment** pass also fires at each interim-adapter rollover (per sub-interval, default 12h) when enough new triples have accumulated since the last pass (`graph_enrichment_min_triples_floor`), amortising the SOTA cost across the 84h cycle instead of concentrating it at the final boundary. Both passes are budget-bound by `graph_enrichment_neighborhood_hops` and `graph_enrichment_max_entities_per_pass`.

**Background training** is driven by a systemd user timer whose period derives from `consolidation.refresh_cadence` (default `"12h"`). The full-consolidation period is derived, not configured: `refresh_cadence × max_interim_count` (default 12h × 7 = 84h). Interim adapters absorb new facts between full cycles so recall does not wait a full period. `BackgroundTrainer` pauses at step boundaries for inference requests, switches the model between eval/train mode, and saves `resume_state.json` + `bg_checkpoint/` at each epoch boundary so a crash mid-cycle resumes from the last completed epoch rather than restarting from zero. A missed post-session training trigger is replayed from a persistent queue (`post_session_queue.json`) on startup. A **simulation mode** (`consolidation.mode: simulate`) runs extraction only, saving results to a debug directory without training.

**Speaker identification** uses pyannote 512-dim voice embeddings with multi-embedding centroid matching and auto-enrichment on confirmed matches.

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
HF_DEACTIVATE_ASYNC_LOAD=1               # required on WSL2 for models >= 4B params

# Server (required for HA integration)
HA_URL=http://<your-ha-ip>:8123          # Home Assistant URL
HA_TOKEN=<your-ha-long-lived-token>      # HA → Profile → Long-Lived Access Tokens
GROQ_API_KEY=<your-groq-api-key>         # groqcloud.com → API Keys
```

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
├── models/           # QLoRA model loading, multi-adapter management
├── training/         # LoRA fine-tuning, replay, consolidation loop
│   ├── indexed_memory.py   # Indexed key recall + SimHash verification
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

### RTX 50-series (Blackwell) Note

**bitsandbytes:** Version 0.49.2 lacks native CUDA kernels for sm_120 (Blackwell architecture). NF4 quantization will crash when loading models larger than ~3B parameters. The fix is to install from the main branch, which includes native sm_120 binaries:

```bash
pip install bitsandbytes --upgrade --pre
# or from source:
pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git
```

This is a build-infrastructure issue (native kernels vs PTX JIT compilation), not a correctness issue — the kernels are functionally identical, just not yet performance-tuned for Blackwell. Once bitsandbytes 0.50.0 is released, the standard `pip install` will work.

**WSL2 threaded weight loading:** Transformers 5.3+ uses a `ThreadPoolExecutor` (4 workers) to parallelize weight loading. On WSL2, concurrent `tensor.to('cuda')` calls from worker threads can race the dxg paravirt memory mapper (`dxgkio_make_resident`), causing `CUDA driver error: device not ready` on models >= 4B parameters. If you encounter this, disable threaded loading:

```bash
# Add to your .env or shell profile
export HF_DEACTIVATE_ASYNC_LOAD=1
```

This forces sequential weight loading. Models load slightly slower but reliably. The issue is specific to WSL2's DirectX Graphics (dxg) layer — native Linux is unaffected.

**WSL2 Modern Standby (laptop GPUs):** Windows Modern Standby can power-cycle the GPU during idle periods, causing TDR BSOD (bugcheck 0x116) if a CUDA workload is active. The `acquire_gpu()` context manager in `experiments/utils/gpu_guard.py` prevents this by holding `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` via a background PowerShell process for the duration of GPU workloads. This is automatic for all experiment scripts that use `acquire_gpu()`. A cooling pad is recommended for sustained workloads — the primary benefit is faster thermal recovery between runs rather than higher sustained clocks, as the TGP is the binding constraint under load.

## How It Works

1. **Extract:** LLM-based graph extraction pulls entities and relations from session text (optionally using a dedicated distillation model for higher quality)
2. **Merge:** Entity resolution deduplicates and aggregates knowledge across sessions
3. **Score:** Composite scoring (PageRank + degree + recurrence + recency) identifies promotion candidates
4. **Assign keys:** Each QA pair gets a unique key (`graph1`, `graph2`, ...) for addressable recall
5. **Train:** LoRA adapters learn the key→QA mapping via chat-template formatted training
6. **Verify:** SimHash registry detects hallucination with continuous confidence scoring
7. **Promote:** Well-reinforced facts move from episodic to semantic adapter
8. **Decay:** Unreinforced facts fade after configurable window

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

`configs/server.yaml` is fully commented — every option has inline docs
explaining its effect, privacy implications, and interaction with other
options. A short map of the top-level sections:

| Section | Purpose |
|---------|---------|
| `cloud_only` | Opt-out of local PM — route every query to the SOTA cloud agent. Security-critical. |
| `headless_boot` | Auto-start the server before any interactive login. Reconciles systemd linger + (WSL) a Windows startup task on every start via `scripts/setup/headless-boot.sh`. |
| `server` | Host, port, VRAM safety margin, auto-reclaim polling. |
| `model` | Base model (`mistral`, `gemma`, `qwen3b`, `gemma4`). |
| `debug`, `snapshot_key` | Privacy mode + Fernet key for encrypted session snapshots. |
| `paths` | Data, sessions, debug, prompts directories. |
| `adapters` | Per-adapter `enabled` / `rank` / `alpha` / `learning_rate` / `target_modules`. |
| `consolidation` | **`refresh_cadence` is the only scheduling knob** (default `"12h"`). Full-cycle period is derived: `refresh_cadence × max_interim_count` (default 12h × 7 = 84h). Also gates the extraction pipeline stages (noise filter, plausibility, anonymization, NER check) and the thermal-throttle quiet-hours policy (`quiet_hours_mode` = `always_on`/`always_off`/`auto` with `start`/`end`). |
| `agents` | SOTA cloud fallback (`sota` + `sota_providers`), HA conversation agent id. |
| `tools.ha` | HA URL, token, language filter, entity allowlist, tool timeout. |
| `sanitization` | PII gate for cloud egress (`off`/`warn`/`block`). |
| `voice` | Voice prompt file, per-speaker greeting cadence. |
| `speaker` | pyannote thresholds, enrollment flow, embedding caps. |
| `stt`, `tts` | Whisper model + Wyoming port; Piper/MMS voices per language. |

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
- **VRAM pre-load validator:** `paramem/server/vram_validator.py` proves base model + main adapters + `max_interim_count` + staging slot + STT + TTS + KV cache fits the configured budget before the server loads any weights. The server refuses to start rather than OOM mid-load.

### GPU Lifecycle

The server shares the GPU with ML workloads:

- **SIGUSR1** → switch to cloud-only mode (model unloaded from VRAM)
- **Auto-reclaim** → periodically checks if GPU is free, reloads model
- **Startup guard** → if GPU is occupied at startup, starts in cloud-only mode
- **`--cloud-only`** → explicit flag to skip model loading

```bash
# Release GPU for a training run
kill -SIGUSR1 $(pidof python -m paramem.server.app)

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
| Orphaned (holder PID dead) | `(orphaned hold by [...] (age 15m) — pstatus --force-local)` (yellow) | `SIGKILL`ed test.  Auto-reclaim has emitted a single WARN and stopped looping. |
| Orphaned (no holder registered) | `(orphaned hold, no holder registered — pstatus --force-local)` (yellow) | `PARAMEM_EXTRA_ARGS` set by legacy caller / manual tinkering. |

Operator recovery is a single command:

```bash
pstatus --force-local
# → POST /gpu/force-local: clears PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_*
#   and, if the running server is in --defer-model, restarts so the
#   next boot loads the model locally.
```

Auto-reclaim **never auto-clears orphans** — by design, visibility over
silent self-healing.  The loop stops on orphan detection and waits for
the operator.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, get a response |
| `/status` | GET | Full operational snapshot — server mode, model id + device, per-adapter specs (`rank`/`alpha`/`lr`/`target_kind`), interim adapter inventory + capacity, speaker embedding backend/model/device, STT/TTS engines, enrolled speakers, pending sessions + orphans + oldest age, consolidating flag + BG trainer state, last consolidation result, schedule + next-run ETA, deferred-mode `hold` block (owner PID + liveness + age + cmd hint) |
| `/consolidate` | POST | Trigger consolidation manually (blocking) |
| `/refresh-ha` | POST | Rebuild the HA entity graph from `/api/states` + `/api/services` |
| `/gpu/force-local` | POST | Clear any `PARAMEM_EXTRA_ARGS=--defer-model` hold and, if this process is in defer mode, restart into local mode.  Called by `pstatus --force-local`.  Idempotent. |

**Chat request:**

```bash
curl -X POST http://localhost:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Where does Marcus work?", "conversation_id": "session1"}'
```

The server buffers conversation turns automatically. On the next consolidation cycle, it extracts knowledge from buffered sessions, merges into the knowledge graph, generates QA pairs, and retrains the adapter.

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
- **Speaker identification:** pyannote 512-dim voice embeddings. Multi-embedding profiles (up to 50 per speaker) with L2-normalized centroid matching for cross-device robustness. Auto-enrichment on confirmed matches. Deferred identity binding keeps anonymous utterances bound to a BPE-stable `Speaker{N}` placeholder until the speaker discloses a name, after which the graph is retro-claimed without a rewrite at training time (name resolves at render).
- **Multilingual TTS:** Piper voices per language with MMS-TTS fallback; language detection on the response text, speaker binding so each speaker's preferred voice persists, routed to media players via HA.
- **Anti-confabulation voice prompt:** a separate system prompt at the voice turn tells the model not to invent facts about the speaker when the parametric memory has nothing to say, and to fall through to the SOTA path cleanly.

## Security

ParaMem encrypts every piece of on-disk infrastructure metadata — registry, knowledge graph, session queue, speaker profiles, backup payloads, HF-Trainer checkpoint shards — at rest. The authoritative operator document is [`SECURITY.md`](SECURITY.md); the short version:

- **Two-identity age X25519 model.** A **daily** identity lives on the host, passphrase-wrapped at `~/.config/paramem/daily_key.age` (mode `0600`). A **recovery** identity is printed once at setup time and stored offline by the operator; only its public recipient (`~/.config/paramem/recovery.pub`) persists on the device. Every on-disk envelope lists both recipients so hardware loss is recoverable from the printed paper alone.
- **Startup posture.** The server emits one of four `SECURITY:` log lines at startup and surfaces `encryption: on|off` on `/status`. Mode mismatches (plaintext alongside encrypted, or a key missing for on-disk ciphertext) refuse startup with an actionable message rather than degrade silently.
- **Required env vars for Security-ON:** `PARAMEM_DAILY_PASSPHRASE` (operator-chosen; unlocks the daily age key). Optional: `PARAMEM_API_TOKEN` (bearer token on all REST endpoints; unset = open LAN posture with a loud startup warning), `PARAMEM_LISTEN_IP` / `PARAMEM_NAS_IP` (scope network exposure).

Key lifecycle is driven by the `paramem generate-key` / `migrate-to-age` / `rotate-daily` / `rotate-recovery` / `restore` commands — see [`SECURITY.md`](SECURITY.md) for the first-run walkthrough, threat model, operator responsibilities, and known limitations.

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

`experiments/dataset_probe.py` runs any supported dataset through the full consolidation pipeline (extract → merge → QA → indexed-key train → recall smoke) and emits identically-shaped per-session diagnostics. Useful for comparing extraction quality across corpora and regression-testing the pipeline end-to-end. Resume-safe; outputs land in `outputs/dataset_probe/{dataset}/{model}/{timestamp}/`.

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
# Run a single test
python experiments/test1_scale_expansion.py --model gemma

# Full-pipeline scaling run (resumable via tpause / tresume)
python experiments/test8_large_scale.py --model mistral
```

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
@article{preusser2026indexed,
  title={Indexed Key Retrieval from LoRA Adapters for Continual Learning},
  author={Preusser, Tobias},
  year={2026},
  note={Preprint}
}
```

## Acknowledgments

Developed with assistance from Claude (Anthropic).

## License

MIT. See [LICENSE](LICENSE).
