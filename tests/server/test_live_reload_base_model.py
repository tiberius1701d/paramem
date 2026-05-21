"""Tests for the VRAM guard on ``_live_reload_base_model``.

The in-process reload primitive must, per the cloud-only VRAM-leak fix:

(a) Look before it leaps — refuse the load when the live GPU budget cannot
    fit the topology, reusing the PRODUCTIVE ``measure_external_vram`` +
    ``enforce_live_budget`` check against the boot-time assessment cached in
    ``_state["topology_assessment"]``. ``measure_external_vram`` snapshots
    device-wide occupancy (nvidia-smi), so an external consumer in another
    WSL distro / on the Windows host is counted.

(b) Fail clean — if a load is attempted and OOMs, release every byte
    ParaMem put on the device so the cloud-only server sits at ~0, rather
    than leaking the partial allocation.

These exercise the function directly (no app lifespan). The release, the
load, and the VRAM check are mocked — the contract under test is the
control flow and the resulting ``_state`` mode/reason, not real CUDA.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _fake_config():
    config = MagicMock()
    config.model_name = "mistral"
    return config


def test_preflight_declines_when_budget_insufficient():
    """Live budget can't fit the topology: load NOT attempted; stays cloud-only.

    The upfront release still runs (so ParaMem holds ~0); reason is set to
    ``insufficient_vram`` so callers distinguish a deferral from a crash.
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "local",
        "cloud_only_reason": None,
        "config": _fake_config(),
        "topology_assessment": MagicMock(name="assessment"),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process") as mock_release,
        patch.object(app_module, "measure_external_vram", return_value=(8 * 2**30, 6 * 2**30)),
        patch.object(
            app_module,
            "enforce_live_budget",
            side_effect=app_module.ConfigurationError("over budget"),
        ),
        patch.object(app_module, "_load_model_into_state") as mock_load,
    ):
        app_module._live_reload_base_model()

        mock_load.assert_not_called()
        mock_release.assert_called_once()  # upfront release only; no cleanup pass
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["cloud_only_reason"] == "insufficient_vram"


def test_preflight_skipped_when_no_boot_assessment():
    """No cached boot assessment (cloud-only / cache miss at boot): skip the
    budget check, defer to the live load gate. measure_external_vram and
    enforce_live_budget are not consulted."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _fake_config(),
        "topology_assessment": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "measure_external_vram") as mock_measure,
        patch.object(app_module, "enforce_live_budget") as mock_enforce,
        patch.object(app_module, "_load_model_into_state") as mock_load,
    ):
        app_module._live_reload_base_model()

        mock_measure.assert_not_called()
        mock_enforce.assert_not_called()
        mock_load.assert_called_once()
        assert app_module._state["mode"] == "local"
        assert app_module._state["cloud_only_reason"] is None


def test_successful_reload_sets_local():
    """Pre-flight passes (budget fits) and the load succeeds → mode local."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _fake_config(),
        "topology_assessment": MagicMock(name="assessment"),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "measure_external_vram", return_value=(8 * 2**30, 1 * 2**30)),
        patch.object(app_module, "enforce_live_budget"),  # no raise == fits
        patch.object(app_module, "_load_model_into_state") as mock_load,
    ):
        app_module._live_reload_base_model()

        mock_load.assert_called_once()
        assert app_module._state["mode"] == "local"
        assert app_module._state["cloud_only_reason"] is None


def test_load_failure_releases_and_stays_cloud_only():
    """Load OOMs after the pre-flight passed: the partial allocation is freed
    (a second release pass) and the server stays cloud-only with reason
    ``reload_failed`` — no leak, no false 'local'."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _fake_config(),
        "topology_assessment": MagicMock(name="assessment"),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process") as mock_release,
        patch.object(app_module, "measure_external_vram", return_value=(8 * 2**30, 1 * 2**30)),
        patch.object(app_module, "enforce_live_budget"),
        patch.object(
            app_module,
            "_load_model_into_state",
            side_effect=RuntimeError("CUDA out of memory"),
        ),
    ):
        app_module._live_reload_base_model()

        # Upfront release + post-failure cleanup release.
        assert mock_release.call_count == 2
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["cloud_only_reason"] == "reload_failed"
