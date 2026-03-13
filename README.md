# ParaMem

**Indexed key retrieval for continual learning in personal LLM agents.**

Knowledge lives in LoRA adapter weights, not in files.

## The Idea

Personal AI agents need persistent memory. Current approaches — RAG, text-based memory, conversation logs — are digital filing cabinets: they store and retrieve text, but the model itself learns nothing. Every session starts from scratch.

ParaMem takes a different approach inspired by biological memory consolidation. Session experiences are extracted into a knowledge graph, converted to QA training pairs, and compressed into LoRA adapter weights through replay-and-consolidation cycles. The model *learns* your facts — they become part of its parameters, not entries in a database.

The core mechanism is **indexed key retrieval**: each fact gets a unique key (`graph1`, `graph2`, ...) and the adapter learns to recall the exact QA pair when prompted with that key. A SimHash registry provides hallucination detection — the system knows what it knows and rejects queries for facts it hasn't learned.

## Key Results

| Test | Result |
|------|--------|
| Per-fact indexed recall | 9/10 exact (rank 8, 30 epochs) |
| Capacity | 19/20 at 20 pairs (95%) |
| Incremental learning | 14/15 after 10+5 addition |
| Two-adapter promotion | Episodic 9/10, Semantic 4/5 |
| Full consolidation (10 cycles) | 17/17 keys, 100% recall both adapters |
| Hallucination detection | 5/5 blocked (SimHash registry) |

All experiments run on a single RTX 5070 (8GB VRAM) using QLoRA 4-bit quantization.

## Architecture

```
                    ┌─────────────────────────┐
                    │     Base Model (3B)      │
                    │   QLoRA 4-bit frozen     │
                    └────┬──────────┬──────────┘
                         │          │
                ┌────────┴──┐  ┌───┴────────┐
                │ Episodic  │  │  Semantic   │
                │ Adapter   │  │  Adapter    │
                │ (rank 8)  │  │  (rank 24)  │
                └─────┬─────┘  └──────┬──────┘
                      │               │
                      └───────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              │      Consolidation Loop       │
              │                               │
              │  extract → merge → score →    │
              │  assign keys → train →        │
              │  promote → decay              │
              └───────────────┬───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Knowledge Graph   │
                    │  (transient layer) │
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

Expected output: 9/10 exact key recall, 0 hallucinations, 5/5 untrained keys blocked.

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
├── graph/            # Knowledge graph extraction, merging, scoring
├── evaluation/       # Recall metrics, embedding scoring, fidelity
└── utils/            # Configuration (YAML-driven)
configs/              # Default configuration
experiments/          # Validated experiment scripts
examples/             # Self-contained example scripts
tests/                # Unit tests (215 tests)
archive/              # Failed approaches (part of the research story)
```

## Hardware Requirements

- **Minimum:** GPU with 8GB VRAM (QLoRA 4-bit quantization)
- **Tested on:** NVIDIA RTX 5070, WSL2, CUDA via conda
- **Base models tested:** Qwen 2.5 3B (primary), Llama 3.2 3B (secondary)
- **Training time:** ~4 min for smoke test (10 keys, 30 epochs)

## How It Works

1. **Extract:** LLM-based graph extraction pulls entities and relations from session text
2. **Merge:** Entity resolution deduplicates and aggregates knowledge across sessions
3. **Score:** Composite scoring (PageRank + degree + recurrence + recency) identifies promotion candidates
4. **Assign keys:** Each QA pair gets a unique key (`graph1`, `graph2`, ...) for addressable recall
5. **Train:** LoRA adapters learn the key→QA mapping via chat-template formatted training
6. **Verify:** SimHash registry detects hallucination with continuous confidence scoring
7. **Promote:** Well-reinforced facts move from episodic to semantic adapter
8. **Decay:** Unreinforced facts fade after configurable window

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
@article{paramem2026,
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
