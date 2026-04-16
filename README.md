# ParaMem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19502523.svg)](https://doi.org/10.5281/zenodo.19502523)

**Indexed key retrieval for continual learning in personal LLM agents.**

Knowledge lives in LoRA adapter weights, not in files.

## Motivation

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — store and retrieve text, but the model itself learns nothing. Every session starts from the same frozen weights.

ParaMem takes a different approach inspired by biological memory consolidation. Session experiences are extracted into a knowledge graph, converted to QA training pairs, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact QA pair when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned.

## Key Results

**What works:**

| Test | Gemma 2 9B | Mistral 7B | Qwen 2.5 3B |
|------|-----------|-----------|-------------|
| Indexed recall at scale | 100/100 | 54/54 | 20/20 |
| Incremental learning (add 5, retrain all) | 15/15 | 15/15 | 15/15 |
| Contradiction resolution (16 fact updates) | 16/16 | 16/16 | — |
| Consolidation loop (10 cycles) | 100% | 100% | 100% |
| PM vs RAG reasoning quality (embedding sim.) | 0.687 vs 0.679 | 0.566 vs 0.525 | — |
| Full replay: recall after 5 add cycles | 44/45 | 45/45 | — |
| Hallucination detection (SimHash registry) | 5/5 blocked | 5/5 blocked | 5/5 blocked |

*— = not tested (Qwen 2.5 3B is a base model without structured output capability, used only for development experiments with pre-defined QA pairs)*

**What doesn't:**

| Test | Gemma 2 9B | Mistral 7B |
|------|-----------|-----------|
| No replay: old key survival after 5 add cycles | 4/40 | 0/40 |
| Adapter composition (additive or merged) | fails | fails |

All experiments run on a single RTX 5070 (8GB VRAM) using QLoRA 4-bit quantization.

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

**Background training** runs on a configurable interval (default: every 2 hours). The `BackgroundTrainer` pauses at step boundaries for inference requests and switches the model between eval/train mode automatically. A **simulation mode** (`consolidation.mode: simulate`) runs extraction only, saving results to a debug directory without training.

**Speaker identification** uses pyannote 512-dim voice embeddings with multi-embedding centroid matching and auto-enrichment on confirmed matches.

## Quick Start

### Requirements

- Python 3.11+
- GPU with 8GB+ VRAM (tested on RTX 5070)
- CUDA toolkit

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
├── utils/            # Shared test harness + PerLTQA dataset loader
├── test1-7_*.py      # Extended evaluation suite (see below)
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
bash scripts/start-server.sh

# Or with systemd (recommended)
systemctl --user enable --now paramem-server

# Verify
curl http://localhost:8420/status
```

The server listens on port 8420. On startup it auto-detects GPU availability — if another process holds the GPU (e.g., a training run), it starts in cloud-only mode and auto-reclaims once the GPU is free.

### Configuration

Edit `configs/server.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8420
  reclaim_interval_minutes: 10

model: mistral          # mistral | gemma | qwen3b | qwen | ministral | llama | gemma4

paths:
  data: data/ha
  sessions: data/ha/sessions
  prompts: configs/prompts

adapters:
  episodic:
    enabled: true
    rank: 8
    alpha: 16

consolidation:
  schedule: "every 2h"  # "HH:MM" (daily) or "every Nh"/"every Nm" (interval); "" = manual only

agents:
  sota:                 # SOTA cloud fallback for reasoning queries
    enabled: true
    provider: anthropic
    model: claude-sonnet-4-6
    api_key: ${ANTHROPIC_API_KEY}
  ha_agent_id: conversation.groq  # HA conversation agent for tool execution

tools:
  ha:
    url: ${HA_URL}
    token: ${HA_TOKEN}
    auto_discover: true
    supported_languages: [en, de, fr, es]
    allowlist:
      - light.*
      - switch.*
      - script.*
      - climate.*
      - media_player.*
  tool_timeout_seconds: 10        # allows for VPN latency
```

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

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, get a response |
| `/status` | GET | Server health, mode, model info, key count |
| `/consolidate` | POST | Trigger consolidation manually |

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
- **Speaker identification:** Pyannote 512-dim voice embeddings. Multi-embedding profiles with L2-normalized centroid matching for cross-device robustness. Auto-enrichment on confirmed matches.
- **TTS to Sonos:** ESPHome voice satellites fire `esphome.tts_uri` events, HA automation routes to room-appropriate Sonos speaker via `media_player.play_media` with `announce: true`.

Tested voice satellites: ReSpeaker Lite (living room), ESP32 S3 Box 3 (office).

## Data

Synthetic test data (`data/synthetic/`) is included in the repository. Two additional datasets are used:

- **Synthetic sessions** (`data/synthetic/synthetic_sessions.json`) — 55 conversational sessions for the end-to-end consolidation loop (§5.5). Included in the repo.
- **PerLTQA** — Public dataset with character profiles and dialogues, used by Tests 1-7 for realistic conversational data. Must be downloaded manually:

```bash
git clone https://github.com/Elvin-Yiming-Du/PerLTQA data/external/PerLTQA
```

Tests fall back to synthetic data if PerLTQA is not available, but results may differ from the paper.

## Extended Evaluation Suite

Seven experiments validating the parametric memory mechanism. Each is a standalone script in `experiments/`.

| Test | What it measures |
|------|------------------|
| `test1_scale_expansion.py` | Recall at 10–100 keys with on-the-fly distillation |
| `test2_contradictions.py` | Contradiction detection and resolution over time |
| `test3_inference.py` | Reasoning quality parity with RAG |
| `test4_reinforcement.py` | Cumulative train-delete-retrain robustness |
| `test5_natural_recall.py` | Keyed vs natural recall gap (motivates key mechanism) |
| `test6_footprint.py` | Storage footprint: adapter vs RAG index at scale |
| `test7_second_persona.py` | Multi-persona generalization + isolation |

```bash
# Run a single test
python experiments/test1_scale_expansion.py --model gemma
```

Results are saved to `outputs/testN_*/{model}/{timestamp}/results.json`.

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
