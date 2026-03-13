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
pytest tests/test_indexed_memory.py -v

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

# Smoke test (~4 min)
python experiments/phase4_indexed_keys_smoke.py --num-epochs 30

# Examples (~4-10 min each)
python examples/quick_start.py
python examples/incremental_learning.py
python examples/two_adapter_promotion.py
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
