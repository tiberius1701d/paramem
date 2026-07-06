# Contributing to ParaMem

## Development Setup

```bash
# Clone the repository
git clone https://github.com/tiberius1701d/paramem.git
cd paramem

# Option 1: pip install (editable)
pip install -e ".[dev]"

# Option 2: conda
conda env create -f environment.yml
conda activate paramem
```

## Running Tests

```bash
# Full test suite (CPU-only, no GPU required)
pytest tests/

# Specific test file
pytest tests/test_entry_format.py -v

# Quick check
pytest tests/ -x -q
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
ruff check paramem/ tests/ experiments/

# Auto-fix lint errors
ruff check --fix paramem/ tests/ experiments/

# Check formatting
ruff format --check paramem/ tests/ experiments/

# Apply formatting
ruff format paramem/ tests/ experiments/
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

Prompts are editable without a code change — every LLM-touching pipeline stage reads its prompt from `configs/prompts/*.txt` (see [Tuning prompts](#tuning-prompts) below).

### Tuning prompts

`scripts/dev/calibrate_prompts.py` drives the `/calibrate/*` endpoints against the running server's local model, so a prompt edit can be validated live (extract, anonymize, enrich, plausibility, and the other calibration stages) without touching training code. The endpoints it calls are gated behind `consolidation.calibrate_endpoint_enabled` (default off) — see [DEPLOYMENT.md](DEPLOYMENT.md) for the full `/calibrate/*` reference.

### Running GPU tests

The default `pytest tests/` run auto-deselects `@pytest.mark.gpu` tests (see `tests/conftest.py`) so the CPU-only suite never touches CUDA. To opt in:

```bash
# Run everything, including GPU-marked tests
pytest --gpu

# Recall tests (train ~30 epochs + probe) require --gpu as well
pytest --gpu --recall
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes
3. Ensure all tests pass: `pytest tests/`
4. Ensure lint is clean: `ruff check paramem/ tests/`
5. Ensure formatting is clean: `ruff format --check paramem/ tests/`
6. Write a clear PR description explaining what and why
7. Submit the PR

## Reporting Issues

When reporting a bug, please include:
- Python version and OS
- GPU model and VRAM
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs
