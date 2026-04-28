"""Structural parity across the three server config files.

  configs/server.yaml.example  - shipped operator template  (tracked)
  tests/fixtures/server.yaml   - stable test fixture        (tracked)
  configs/server.yaml          - operator-local override    (gitignored)

The shipped template defines the canonical option surface. The test
fixture mirrors that surface key-for-key with values overridden for
deterministic test runs (Mistral 7B pinned, services disabled, ports
shifted). The operator's local config may add or omit keys; we only
warn when it accumulates orphan keys that the loader cannot consume.

Two checks:

* test_example_and_fixture_have_same_options
    CI-required. If the example gains a key, the fixture must follow
    (with a value appropriate to the test context). If the fixture
    accumulates a key the example doesn't carry, the test surface has
    drifted from the operator-facing surface — also caught.

* test_operator_yaml_keys_subset_of_example
    Developer-only (skipped in CI when configs/server.yaml is absent).
    Catches keys an operator pinned locally that the example no longer
    ships — those keys are silently ignored by the loader and probably
    represent a stale local override.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

EXAMPLE = Path("configs/server.yaml.example")
FIXTURE = Path("tests/fixtures/server.yaml")
OPERATOR = Path("configs/server.yaml")


def _leaf_keys(yaml_path: Path) -> set[str]:
    """Flatten a nested YAML into the set of dotted-path leaf keys.

    ``a.b.c`` for scalars; lists are leaves at the parent path.
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}

    def _walk(node, prefix: str) -> set[str]:
        if not isinstance(node, dict):
            return {prefix}
        out: set[str] = set()
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else key
            out |= _walk(value, child)
        return out

    return _walk(data, "")


def test_example_and_fixture_have_same_options():
    """Strict key-set equality — values may differ, but the surface must match."""
    example_keys = _leaf_keys(EXAMPLE)
    fixture_keys = _leaf_keys(FIXTURE)

    only_example = example_keys - fixture_keys
    only_fixture = fixture_keys - example_keys

    diagnostic_lines = []
    if only_example:
        diagnostic_lines.append(
            f"Keys only in {EXAMPLE} (fixture is missing them):\n  "
            + "\n  ".join(sorted(only_example))
        )
    if only_fixture:
        diagnostic_lines.append(
            f"Keys only in {FIXTURE} (example is missing them):\n  "
            + "\n  ".join(sorted(only_fixture))
        )

    assert not (only_example or only_fixture), (
        "configs/server.yaml.example and tests/fixtures/server.yaml diverged "
        "in option surface. When you add a new field to one, add it to the "
        "other (with a value appropriate to that file's role).\n\n" + "\n\n".join(diagnostic_lines)
    )


@pytest.mark.skipif(
    not OPERATOR.exists(),
    reason="operator-local configs/server.yaml absent (CI / fresh clone)",
)
def test_operator_yaml_keys_subset_of_example():
    """Developer-only. Catches keys the operator pinned that the example
    no longer ships — those keys are silently ignored by the loader and
    probably represent a stale local override.
    """
    operator_keys = _leaf_keys(OPERATOR)
    example_keys = _leaf_keys(EXAMPLE)
    orphans = operator_keys - example_keys

    assert not orphans, (
        f"configs/server.yaml has keys not in {EXAMPLE}:\n  "
        + "\n  ".join(sorted(orphans))
        + "\n\nThese are silently ignored by the loader. Either remove "
        "them from your local server.yaml, or land them in the example "
        "first."
    )
