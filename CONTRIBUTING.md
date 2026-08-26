# Contributing to ParaMem

## Development Setup

```bash
# Clone the repository
git clone https://github.com/tiberius1701d/paramem.git
cd paramem

# Option 1: pip install (editable)
pip install -e ".[dev]"

# Option 2: conda (the env file provides the interpreter; the package installs with pip)
conda env create -f environment.yml
conda activate paramem
pip install -e ".[dev]"
```

## Running Tests

```bash
# What CI runs (CPU only, per-test timeout enforced)
pytest tests/ -v --timeout=60 -m "not gpu"

# Specific test file
pytest tests/test_entry_format.py -v

# Quick check
pytest tests/ -x -q
```

The `integration`-marked tests shell out to the installed `paramem` console script and skip unless it is found; set `PARAMEM_BINARY` to your environment's copy to run them.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
ruff check paramem/ tests/ experiments/ scripts/

# Auto-fix lint errors
ruff check --fix paramem/ tests/ experiments/ scripts/

# Check formatting
ruff format --check paramem/ tests/ experiments/ scripts/

# Apply formatting
ruff format paramem/ tests/ experiments/ scripts/
```

Configuration is in `pyproject.toml`:
- Line length: 100
- Target: Python 3.11
- Rules: E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort)

## Running Experiments

Experiments require a GPU with 8GB+ VRAM:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smoke test — trains a tiny procedural adapter (attention + MLP target modules)
# and asserts loss drops without CUDA OOM on 8 GB VRAM
python experiments/smoke_procedural_mlp.py

# Post-install REST smoke (needs a running server, debug: true, PARAMEM_API_TOKEN)
python examples/quick_start.py
```

Prompts are editable without a code change — every LLM-touching pipeline stage reads its prompt from `configs/prompts/` (see [Tuning prompts](#tuning-prompts) below). One exception: `configs/prompts/trained_recall.txt` is the exact text every adapter was trained on. A test pins it; changing it means retraining every adapter and updating the pin in the same change.

### Tuning prompts

`scripts/dev/calibrate_prompts.py` drives the `/calibrate/*` endpoints against a running server, so a prompt edit can be validated live without touching training code. `enrich`, `plausibility`, and any `extract` run not stopped before enrichment need cloud egress and are refused without it. `normalize` places a real, billed provider call when cloud egress is permitted and runs on the local model otherwise. `respond` runs a full serving turn: it can actuate a Home Assistant device and place a billed cloud call. The endpoints are gated behind `consolidation.calibrate_endpoint_enabled` (default off) — see [DEPLOYMENT.md](DEPLOYMENT.md#api) for the full `/calibrate/*` reference.

### Running GPU tests

The default `pytest tests/` run auto-deselects `@pytest.mark.gpu` tests (see `tests/conftest.py`) so the CPU-only suite never touches CUDA. To opt in:

```bash
# Run everything, including GPU-marked tests
pytest --gpu

# Run only GPU-marked tests
pytest -m gpu
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes
3. Ensure all tests pass: `pytest tests/ -v --timeout=60 -m "not gpu"`
4. Ensure lint is clean: `ruff check paramem/ tests/ experiments/ scripts/`
5. Ensure formatting is clean: `ruff format --check paramem/ tests/ experiments/ scripts/`
6. Write a clear PR description explaining what and why
7. Submit the PR

## Reporting Issues

Open a GitHub issue using the Bug Report or Feature Request template and fill in every field — the environment section in particular. A bug without steps to reproduce and the exact package versions usually cannot be acted on.
