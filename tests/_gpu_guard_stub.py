"""Shared stub for the ``gpu_guard`` lab-tools dependency.

``gpu_guard`` is a separate lab-tools repository this project does not
depend on for tests to run. Patching ``experiments.utils.gpu_guard.acquire_gpu``
by name would first import the real wrapper module, which itself does
``from gpu_guard import ...`` and fails wherever the package is not
installed. :func:`stub_gpu_guard` inserts both ``gpu_guard`` and
``experiments.utils.gpu_guard`` into ``sys.modules`` directly, sidestepping
that import entirely, so every test that exercises a script depending on
``experiments.utils.gpu_guard.acquire_gpu`` can use one implementation.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest


def stub_gpu_guard(monkeypatch: pytest.MonkeyPatch, acquire_gpu=contextlib.nullcontext) -> None:
    """Replace ``gpu_guard`` and its paramem wrapper with stand-ins; the
    stub's ``acquire_gpu`` defaults to a no-op context manager.

    Args:
        monkeypatch: The test's ``pytest.MonkeyPatch`` fixture (governs undo).
        acquire_gpu: Replacement for ``experiments.utils.gpu_guard.acquire_gpu``.
            Defaults to ``contextlib.nullcontext`` (a no-op context manager);
            a caller that needs to observe or control the acquire call passes
            its own callable.
    """
    guard = types.ModuleType("gpu_guard")
    guard.GPUConfigMissing = type("GPUConfigMissing", (Exception,), {})
    wrapper = types.ModuleType("experiments.utils.gpu_guard")
    wrapper.acquire_gpu = acquire_gpu
    monkeypatch.setitem(sys.modules, "gpu_guard", guard)
    monkeypatch.setitem(sys.modules, "experiments.utils.gpu_guard", wrapper)
