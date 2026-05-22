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


# ---------------------------------------------------------------------------
# Value-parity gate (CI-required)
#
# The structural test above guards key drift.  It does NOT guard *value*
# drift — and that's exactly how `extraction_max_tokens` was 8192 in
# production but 2048 in the test fixture for months without anyone
# noticing (until the simulate-mode probe surfaced the truncation in May
# 2026).  This second gate enforces VALUE equality on every leaf, with
# an explicit allowlist of paths where divergence is intentional.
#
# The allowlist is the single, reviewable decision surface for "this
# field is allowed to diverge between the shipped operator default and
# the test fixture's posture".  Adding a path requires writing why.
# ---------------------------------------------------------------------------


_ALLOWED_VALUE_DIVERGENCE = frozenset(
    {
        # --- Test-mode posture: fixture intentionally exercises code paths
        # that the example ships disabled for ship-safety. ----------------
        "agents.sota.enabled",  # cloud agent ON in tests, OFF in shipped default
        "agents.sota_providers.anthropic.enabled",
        "agents.sota_providers.google.enabled",
        "agents.sota_providers.openai.enabled",
        "consolidation.extraction_noise_filter",  # SOTA noise filter ON in tests
        "consolidation.graph_enrichment_enabled",  # graph-tier SOTA ON in tests
        "consolidation.graph_enrichment_interim_enabled",
        # ON in tests + live deployment; example ships OFF (default-OFF rollout posture).
        "consolidation.recall_early_stopping",
        # MMS/VITS TTS on GPU in tests + live deployment (worth the VRAM); example
        # ships all-CPU for ship-safety. Piper voices inherit the cpu default in all three.
        "tts.voices.tl.device",
        "debug",  # tests want full diagnostic artefacts
        "sanitization.cloud_mode",  # tests anonymize-and-send; example blocks
        # --- Sandbox-rooted paths: every path the fixture writes to must
        # live under tests/fixtures/sandbox/ so a test run can't clobber
        # production data. ------------------------------------------------
        "paths.data",
        "paths.debug",
        "paths.sessions",
    }
)


def _leaf_items(yaml_path: Path) -> dict[str, object]:
    """Walk YAML into a dict of dotted-path → leaf value."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}

    out: dict[str, object] = {}

    def _walk(node, prefix: str) -> None:
        if not isinstance(node, dict):
            out[prefix] = node
            return
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else key
            _walk(value, child)

    _walk(data, "")
    return out


def test_pipeline_critical_values_match():
    """Every leaf must hold the same value in example and fixture, except
    for paths in :data:`_ALLOWED_VALUE_DIVERGENCE`.

    This is the gate that catches drifts like
    ``extraction_max_tokens=2048`` in the fixture vs ``8192`` in the
    example — the kind of drift that silently truncates the SOTA
    extraction chain during simulate-mode prompt-engineering iteration
    and produces lower-quality output than production.

    To allow a new divergence: add the path to
    :data:`_ALLOWED_VALUE_DIVERGENCE` with a comment explaining why.
    """
    example = _leaf_items(EXAMPLE)
    fixture = _leaf_items(FIXTURE)

    shared_keys = example.keys() & fixture.keys()
    diffs = {
        path: (example[path], fixture[path])
        for path in shared_keys
        if example[path] != fixture[path]
    }
    unexpected = {
        path: vals for path, vals in diffs.items() if path not in _ALLOWED_VALUE_DIVERGENCE
    }

    if unexpected:
        side_by_side = "\n".join(
            f"  {path}\n    {EXAMPLE.name}: {ex_val!r}\n    {FIXTURE.name}: {fx_val!r}"
            for path, (ex_val, fx_val) in sorted(unexpected.items())
        )
        raise AssertionError(
            f"Unexpected value drift between {EXAMPLE} and {FIXTURE}:\n\n"
            f"{side_by_side}\n\n"
            "Either align the two files (preferred) or, if the divergence "
            "is intentional, add the path to _ALLOWED_VALUE_DIVERGENCE in "
            f"{Path(__file__).relative_to(Path.cwd())} with a one-line "
            "comment justifying why the values must differ."
        )


def test_allowlist_does_not_cover_paths_that_already_match():
    """Hygiene: an allowlist entry that no longer corresponds to an
    actual divergence is dead weight — the original reason for the
    entry is gone.  Drop it so the next reviewer doesn't have to
    re-derive intent.
    """
    example = _leaf_items(EXAMPLE)
    fixture = _leaf_items(FIXTURE)

    matching_paths = {
        path for path in example.keys() & fixture.keys() if example[path] == fixture[path]
    }
    stale = sorted(_ALLOWED_VALUE_DIVERGENCE & matching_paths)

    assert not stale, (
        "These paths are listed in _ALLOWED_VALUE_DIVERGENCE but their "
        "values now match in both files — the divergence was resolved "
        "and the allowlist entry is stale:\n  "
        + "\n  ".join(stale)
        + "\n\nRemove these entries from _ALLOWED_VALUE_DIVERGENCE."
    )


def test_allowlist_only_lists_real_paths():
    """Hygiene: an allowlist entry pointing at a path that doesn't exist
    in either file is a typo or a stale reference to a removed setting.
    """
    example = _leaf_items(EXAMPLE)
    fixture = _leaf_items(FIXTURE)
    known_paths = example.keys() | fixture.keys()
    phantom = sorted(_ALLOWED_VALUE_DIVERGENCE - known_paths)

    assert not phantom, (
        "_ALLOWED_VALUE_DIVERGENCE references paths that don't exist "
        f"in either {EXAMPLE} or {FIXTURE}:\n  "
        + "\n  ".join(phantom)
        + "\n\nFix the typo or remove the stale entry."
    )
