"""Lint: which config loader is each test allowed to use?

Rule:

* GPU / contract / integration tests load the model via
  ``load_server_config("tests/fixtures/server.yaml")``. The fixture pins
  Mistral 7B and disables external-dep services so calibration is stable.

* ``configs/server.yaml.example`` is only loaded by tests that explicitly
  verify the shipped operator template (allowlist below). Loading the
  example as a model fixture would drift as deployment patterns evolve and
  break calibrated thresholds.

If you hit this lint:
  * You probably copied a fixture from another contract test. The pattern
    other tests should mirror is::

        from paramem.server.config import load_server_config
        from paramem.models.loader import load_base_model

        cfg = load_server_config("tests/fixtures/server.yaml")
        model, tokenizer = load_base_model(cfg.model_config, cfg.tier_config_map())

  * If your test verifies the example file itself, add it to the allowlist
    below and document why in a comment.
"""

from __future__ import annotations

import re
from pathlib import Path

# Tests that legitimately load configs/server.yaml.example because their
# purpose IS to verify the shipped template. New entries here must be
# justified in a comment.
EXAMPLE_VERIFY_ALLOWLIST = frozenset(
    {
        # Smoke test for the example itself — disabled-by-default invariants.
        "tests/server/test_server_yaml_example.py",
        # Parity test compares example ↔ fixture key sets.
        "tests/server/test_config_parity.py",
        # Validates that the shipped example loads under all dataclass
        # validators (sanitization.cloud_mode, etc.).
        "tests/server/test_config.py",
    }
)

_TESTS_ROOT = Path("tests")


def _iter_test_files() -> list[Path]:
    """All Python files under tests/, except __pycache__ and this lint itself."""
    return [
        p
        for p in _TESTS_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "test_test_config_loader_usage.py"
    ]


def test_no_test_loads_example_yaml_outside_allowlist():
    """A test loading configs/server.yaml.example as its model fixture would
    drift as the example evolves. Use tests/fixtures/server.yaml instead.
    """
    pattern = re.compile(r'load_server_config\(\s*["\']configs/server\.yaml\.example["\']')
    violations: list[str] = []

    for py in _iter_test_files():
        rel = py.as_posix()
        if rel in EXAMPLE_VERIFY_ALLOWLIST:
            continue
        text = py.read_text()
        for line_num, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{rel}:{line_num}: {line.strip()}")

    assert not violations, (
        "Tests must not load configs/server.yaml.example as a model fixture "
        "(it drifts as deployment patterns evolve, breaking calibrated "
        "thresholds). Use load_server_config('tests/fixtures/server.yaml') "
        "instead.\n\nViolations:\n  " + "\n  ".join(violations)
    )
