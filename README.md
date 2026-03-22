# ParaMem

**Indexed key retrieval for continual learning in personal LLM agents.**

Knowledge lives in LoRA adapter weights, not in files.

## The Idea

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — store and retrieve text, but the model itself learns nothing. Every session starts from the same frozen weights.

ParaMem takes a different approach inspired by biological memory consolidation. Session experiences are extracted into a knowledge graph, converted to QA training pairs, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact QA pair when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned.

## Key Results

| Test | Result |
|------|--------|
| Per-fact indexed recall | 10/10 exact (rank 8, 30 epochs) |
| Capacity | 20/20 at 20 pairs (100%) |
| Incremental learning | 15/15 after 10+5 addition |
| Two-adapter promotion | Episodic 10/10, Semantic 5/5 |
| Full consolidation (10 cycles) | 100% recall both adapters |
| Hallucination detection | 5/5 blocked (SimHash registry) |

All experiments run on a single RTX 5070 (8GB VRAM) using QLoRA 4-bit quantization.

## Architecture

```
               ┌──────────────────────────┐
               │        Base Model        │
               │    QLoRA 4-bit frozen    │
               └─────┬────────────┬───────┘
                     │            │
            ┌────────┴───┐  ┌─────┴────────┐
            │  Episodic  │  │   Semantic   │
            │  Adapter   │  │   Adapter    │
            │  (rank 8)  │  │   (rank 24)  │
            └─────┬──────┘  └──────┬───────┘
                  │                │
                  └───────┬────────┘
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

**Episodic adapter** holds recent facts with indexed keys for per-fact retrieval. **Semantic adapter** holds promoted, well-reinforced knowledge. The knowledge graph is a transient processing layer — like the visual cortex, it structures input but doesn't store long-term memory. The adapters are the memory.

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

### Run the Smoke Test

```bash
# Set VRAM allocation strategy (recommended for 8GB GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
tests/                # Unit tests (243 tests)
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

ParaMem includes a REST server for persistent deployment. The server keeps the model loaded in VRAM, serves chat inference, and runs daily consolidation to learn from new conversations.

### Quick Start

```bash
# Start the server (foreground)
bash scripts/start-server.sh

# Or background with log file
bash scripts/start-server.sh --background
```

The server listens on port 8420 by default. Verify it's running:

```bash
curl http://localhost:8420/status
```

### Configuration

Edit `configs/server.yaml`:

```yaml
model: gemma          # gemma | mistral
adapter_dir: data/ha/adapters
registry_path: data/ha/registry.json
graph_path: data/ha/graph.json
session_dir: data/ha/sessions

training:
  epochs: 30
  rank: 8

consolidation:
  schedule: "02:00"   # daily at 2am (empty string = manual only)
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, get a response. Adapter always active. |
| `/status` | GET | Server health, model info, key count, consolidation state |
| `/consolidate` | POST | Trigger consolidation manually |

**Chat request:**

```bash
curl -X POST http://localhost:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Where does Marcus work?", "conversation_id": "session1"}'
```

The server buffers conversation turns automatically. On the next consolidation cycle, it extracts knowledge from buffered sessions, merges into the knowledge graph, generates QA pairs, and retrains the adapter.

### Systemd Service

For persistent deployment:

```bash
sudo cp scripts/paramem-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now paramem-server

# Check logs
journalctl -u paramem-server -f
```

### Home Assistant Integration

A custom conversation agent for Home Assistant is included in `custom_components/paramem/`. Copy or symlink to your HA `custom_components/` directory and configure via the HA UI (Settings → Devices → Add Integration → ParaMem). The component is a thin REST client — all inference runs on the ParaMem server.

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
