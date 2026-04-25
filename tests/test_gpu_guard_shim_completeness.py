"""Drift detector: asserts every test-patched internal name resolves on the shim.

When a name is patched in test_gpu_guard_unknown_approval.py or
test_sleep_inhibitor.py, it is patched at the shim module so that the original
paramem tests that reference ``experiments.utils.gpu_guard.<name>`` continue to
work without edits.  This test enumerates those names and fails fast if any of
them drift (renamed, removed, or forgotten in a future refactor).

When the shim is eventually deleted (V3+), this test deletes with it.
"""

from __future__ import annotations

import importlib

import pytest

# Names that paramem tests patch via ``experiments.utils.gpu_guard.<name>``.
# These must all resolve (via re-export or definition) on the shim module.
_PATCHED_INTERNALS = [
    "_systemd_main_pid",
    "_listener_pid",
    "_pid_cmdline",
    "_is_paramem_server_cmdline",
    "_identify_server_pids",
    "_SleepInhibitor",
    "notify_ml",
    "_get_gpu_pids",
]


@pytest.mark.parametrize("name", _PATCHED_INTERNALS)
def test_patched_internal_resolves(name: str) -> None:
    """Each name in _PATCHED_INTERNALS must be importable from the shim module."""
    shim = importlib.import_module("experiments.utils.gpu_guard")
    obj = getattr(shim, name, None)
    assert obj is not None, (
        f"'{name}' is not present on experiments.utils.gpu_guard. "
        "Update the shim re-exports or this list."
    )
