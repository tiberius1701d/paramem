"""Lint: which config loader is each test allowed to use?

Rule:

* GPU / contract / integration tests load the model via
  ``load_server_config("tests/fixtures/server.yaml")``. The fixture pins
  Mistral 7B and disables external-dep services so calibration is stable.

* ``configs/server.yaml.example`` is only loaded by tests that explicitly
  verify the shipped operator template (allowlist below). Loading the
  example as a model fixture would drift as deployment patterns evolve and
  break calibrated thresholds.

* ``load_config()`` (from ``paramem.utils.config``) is forbidden in tests/.
  It reads ``configs/default.yaml`` which pins Qwen 2.5 3B base — unsuitable
  for any test that exercises structured-output generation. Use
  ``load_server_config("tests/fixtures/server.yaml")`` instead.

If you hit this lint:
  * You probably copied a fixture from another contract test. The pattern
    other tests should mirror is::

        from paramem.server.config import load_server_config
        from paramem.models.loader import load_base_model

        cfg = load_server_config("tests/fixtures/server.yaml")
        model, tokenizer = load_base_model(cfg.model_config)

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
        # validators (role_aware_grounding, sanitization.cloud_mode, etc.).
        "tests/server/test_config.py",
    }
)

# Tests still using paramem.utils.config.load_config(). The default.yaml
# retirement arc removes entries here as each test pivots to
# load_server_config("tests/fixtures/server.yaml"). Only the loader's own
# meta-test remains: tests/test_config.py exercises load_config() directly
# and stays here until the loader retires entirely (Step 7 of the arc).
LEGACY_LOAD_CONFIG_ALLOWLIST = frozenset(
    {
        "tests/test_config.py",
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


def test_no_test_uses_legacy_load_config():
    """The training-side ``load_config()`` reads configs/default.yaml which
    pins Qwen 2.5 3B base — unsuitable for tests that exercise structured-
    output generation, and slated for retirement.

    Use ``load_server_config("tests/fixtures/server.yaml").model_config``.
    """
    pattern = re.compile(r"\bload_config\s*\(")
    violations: list[str] = []

    for py in _iter_test_files():
        rel = py.as_posix()
        if rel in LEGACY_LOAD_CONFIG_ALLOWLIST:
            continue
        text = py.read_text()
        # Skip files that don't import load_config from the training-side module.
        if "from paramem.utils.config import" not in text and "paramem.utils.config" not in text:
            continue
        if "load_config" not in text:
            continue
        for line_num, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{rel}:{line_num}: {line.strip()}")

    assert not violations, (
        "Tests must not call paramem.utils.config.load_config() — it reads "
        "configs/default.yaml (Qwen 2.5 3B base, unsuitable for tests that "
        "need structured-output generation; slated for retirement). Use "
        "load_server_config('tests/fixtures/server.yaml').model_config "
        "instead.\n\nViolations:\n  " + "\n  ".join(violations)
    )
