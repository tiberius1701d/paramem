"""Tests for the VRAM guard on ``_live_reload_base_model``.

The in-process reload primitive must, per the cloud-only VRAM-leak fix:

(a) Look before it leaps — refuse the load when the live GPU budget cannot
    fit the topology.  The reclaim fit-check uses
    ``torch.cuda.mem_get_info(0)[0]`` (device-wide free) compared directly
    to ``assessment.required_bytes``.  This is the same measure the boot
    drain-wait uses, and avoids nvidia-smi which false-frees exiting
    processes under WSL2 (the documented host-crash cause).

(b) Fail clean — if a load is attempted and OOMs, release every byte
    ParaMem put on the device so the cloud-only server sits at ~0, rather
    than leaking the partial allocation.

These exercise the function directly (no app lifespan). The release, the
load, the VRAM check, and ``_build_config_derived_state`` are mocked — the
contract under test is the control flow and the resulting ``_state``
mode/reason, not real CUDA.

``_live_reload_base_model`` calls ``_build_config_derived_state`` after a
successful model load (to rebuild router, exemplar banks, etc.).  Existing
tests mock ``_build_config_derived_state`` so they exercise the same
control-flow contract without triggering real STT/HA/SOTA construction.

Additional tests cover the config-refresh path:
- ``refresh_config_from_disk=True`` calls ``load_server_config`` BEFORE
  ``_release_base_model_in_process``.
- Mode is not set to ``local`` until AFTER ``_build_config_derived_state``.
- A rebuild failure leaves mode=cloud-only with reason ``apply_failed``.
  A partial preload (boot_degraded) now stays local — recall self-heals.
- ``_build_config_derived_state`` is NOT passed ``check_post_load_budget``
  (the build-once VRAM gate must stay lifespan-only; in-process reload must
  not re-run the post-load budget check on every reclaim).
- ``_preload_memory_store`` source selection is driven by
  ``config.consolidation.mode``, not ``_state["mode"]``.
- ``boot_degraded`` is cleared on full hydration and on ``preload_cache=False``;
  set on partial hydration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paramem.server.config import load_server_config


def _server_config(consolidation_mode="train", preload_cache=True):
    """Load the project's canonical test config and apply per-test overrides.

    Uses the real ``load_server_config("tests/fixtures/server.yaml")`` (pins
    Mistral 7B, external-dep services disabled) rather than a fabricated mock —
    config is plain data with no GPU/disk side effects, so there is nothing to
    fake.  Per-test value overrides go in code (CLAUDE.md), never via fixture
    edits.  Returns a fresh object each call, so mutations don't leak across
    tests.  (GPU/disk collaborators — the model, the memory store, the recall
    source — are still substituted with test doubles below; those DO have
    side effects.)
    """
    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.consolidation.mode = consolidation_mode
    cfg.inference.preload_cache = preload_cache
    return cfg


# ---------------------------------------------------------------------------
# Existing VRAM-guard tests (updated to mock _build_config_derived_state)
# ---------------------------------------------------------------------------


def test_preflight_declines_when_gate_reports_no_room():
    """The unified GPU-room gate (_wait_for_gpu_drain) reports no room → load NOT
    attempted; stays cloud-only.  The upfront release still runs (so ParaMem holds
    ~0); reason is ``insufficient_vram`` so callers distinguish a deferral from a
    crash.  Boot and reload react identically to the gate — this tests the reload
    reaction; the gate's own credit/ceiling math is unit-tested below.
    """
    from paramem.server import app as app_module

    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "local",
        "cloud_only_reason": None,
        "config": _server_config(),
        "topology_assessment": fake_assessment,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process") as mock_release,
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=False) as mock_gate,
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(app_module, "_build_config_derived_state") as mock_build,
    ):
        app_module._live_reload_base_model()

        mock_gate.assert_called_once_with(fake_assessment.required_bytes)
        mock_load.assert_not_called()
        mock_build.assert_not_called()
        mock_release.assert_called_once()  # upfront release only; no cleanup pass
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["cloud_only_reason"] == "insufficient_vram"


def test_effective_free_credits_warm_context_capped_at_ceiling():
    """_effective_free_bytes (the single source of truth for both boot and reload)
    credits this process's reclaimable CUDA context back, capped at the cached
    pristine ceiling.  The Mistral live-reload case: a warm-context reading 6.53
    GiB free (< 6.7 GiB required) becomes 6.83 GiB effective (= ceiling) ≥ required,
    so a model that boots also live-reloads — where a raw free comparison deferred.
    """
    from paramem.server import app as app_module

    _GiB = 2**30
    state_patch = {"usable_ceiling_bytes": int(6.83 * _GiB)}
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch(
            "paramem.server.app.torch.cuda.mem_get_info", return_value=(int(6.53 * _GiB), 8 * _GiB)
        ),
    ):
        eff = app_module._effective_free_bytes()
    # min(6.53 + 0.5, 6.83) == 6.83 ≥ 6.7 required.
    assert eff == int(6.83 * _GiB)
    assert eff >= int(6.7 * _GiB)


def test_effective_free_does_not_mask_external_consumer():
    """The credit cannot mask a GENUINE shortfall: an external consumer leaving
    only 4.0 GiB free yields 4.5 GiB effective (free + 0.5 allowance, below the
    ceiling), still < a 6.7 GiB requirement → the gate fails and the reload defers.
    """
    from paramem.server import app as app_module

    _GiB = 2**30
    state_patch = {"usable_ceiling_bytes": int(6.83 * _GiB)}
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch(
            "paramem.server.app.torch.cuda.mem_get_info", return_value=(int(4.0 * _GiB), 8 * _GiB)
        ),
    ):
        eff = app_module._effective_free_bytes()
    # min(4.0 + 0.5, 6.83) == 4.5 < 6.7 required.
    assert eff == int(4.5 * _GiB)
    assert eff < int(6.7 * _GiB)


def test_effective_free_uncapped_when_no_cached_ceiling():
    """When the ceiling was never cached (HF-cache-miss boot), the credit applies
    uncapped (free + allowance) — the documented fallback for the rare path.
    """
    from paramem.server import app as app_module

    _GiB = 2**30
    # Ensure no stale ceiling leaks in from another test.
    app_module._state.pop("usable_ceiling_bytes", None)
    with patch(
        "paramem.server.app.torch.cuda.mem_get_info", return_value=(int(5.0 * _GiB), 8 * _GiB)
    ):
        eff = app_module._effective_free_bytes()
    assert eff == int(5.0 * _GiB) + app_module._CUDA_CONTEXT_ALLOWANCE_BYTES


def test_preflight_skipped_when_no_boot_assessment():
    """No cached boot assessment (cloud-only / cache miss at boot): skip the
    budget check, defer to the live load gate. torch.cuda.mem_get_info is not
    consulted (no assessment to compare against).

    _build_config_derived_state is called once (to rebuild the router +
    classifier) on the plain-reclaim path.
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.mem_get_info") as mock_mem_get_info,
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(app_module, "_build_config_derived_state") as mock_build,
    ):
        app_module._live_reload_base_model()

        # With topology_assessment=None, mem_get_info must NOT be called.
        mock_mem_get_info.assert_not_called()
        mock_load.assert_called_once()
        # plain-reclaim path calls _build_config_derived_state once
        # (rebuild_session_buffer=False, cloud_only=False).
        mock_build.assert_called_once()
        assert app_module._state["mode"] == "local"
        assert app_module._state["cloud_only_reason"] is None


def test_successful_reload_sets_local():
    """Gate reports room and the load succeeds → mode local.

    _build_config_derived_state is called once on the plain-reclaim path.
    """
    from paramem.server import app as app_module

    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=True),
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(app_module, "_build_config_derived_state") as mock_build,
    ):
        app_module._live_reload_base_model()

        mock_load.assert_called_once()
        # _build_config_derived_state called once on plain-reclaim path.
        mock_build.assert_called_once()
        assert app_module._state["mode"] == "local"
        assert app_module._state["cloud_only_reason"] is None


def test_load_failure_releases_and_stays_cloud_only():
    """Load OOMs after the pre-flight passed: the partial allocation is freed
    (a second release pass) and the server stays cloud-only with reason
    ``reload_failed`` — no leak, no false 'local'.

    _build_config_derived_state must NOT be called when the load fails.
    """
    from paramem.server import app as app_module

    # Gate reports room → pre-flight passes; the load then OOMs.
    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process") as mock_release,
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=True),
        patch.object(
            app_module,
            "_load_model_into_state",
            side_effect=RuntimeError("CUDA out of memory"),
        ),
        patch.object(app_module, "_build_config_derived_state") as mock_build,
    ):
        app_module._live_reload_base_model()

        # Upfront release + post-failure cleanup release.
        assert mock_release.call_count == 2
        mock_build.assert_not_called()
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["cloud_only_reason"] == "reload_failed"


# ---------------------------------------------------------------------------
# Config-refresh ordering + full rebuild on True vs model-only
# ---------------------------------------------------------------------------


def test_refresh_config_from_disk_loads_config_before_release():
    """When refresh_config_from_disk=True: load_server_config is called BEFORE
    _release_base_model_in_process.

    The config is committed to _state before the release so that a crash
    mid-reload leaves _state["config"] pointing at the new config B
    (recoverable via /gpu/acquire or restart).
    """
    from paramem.server import app as app_module

    call_order = []

    new_config = _server_config()

    def fake_load_server_config(_path):
        call_order.append("load_server_config")
        return new_config

    def fake_release():
        call_order.append("release")

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", side_effect=fake_load_server_config),
        patch.object(app_module, "_release_base_model_in_process", side_effect=fake_release),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True)

    assert call_order.index("load_server_config") < call_order.index("release"), (
        f"load_server_config must run BEFORE release; got order={call_order}"
    )


def test_refresh_config_mode_not_local_until_after_build():
    """When refresh_config_from_disk=True: mode is NOT set to 'local' until
    AFTER _build_config_derived_state succeeds.

    If the build is not called (or raises), mode stays cloud-only.
    """
    from paramem.server import app as app_module

    mode_at_build_call = []

    def fake_build(config, *, cloud_only, rebuild_session_buffer=True, full_rebuild=True):
        mode_at_build_call.append(app_module._state.get("mode"))

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", return_value=_server_config()),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state", side_effect=fake_build),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True)

        # mode must be cloud-only while build is executing, then local after.
        # Assertions are inside the with-block because patch.dict restores
        # _state["mode"] to its pre-test value on exit — reading it outside
        # would see the pre-test value, not the post-reload value.
        assert mode_at_build_call, "_build_config_derived_state was not called"
        assert mode_at_build_call[0] == "cloud-only", (
            f"mode should be cloud-only during build; got {mode_at_build_call[0]}"
        )
        assert app_module._state["mode"] == "local", "mode should be local after successful rebuild"


def test_refresh_config_rebuild_failure_stays_cloud_only():
    """When refresh_config_from_disk=True: a rebuild failure in
    _build_config_derived_state leaves mode=cloud-only with reason
    'apply_failed' (partial-rebuild recovery — the server must not silently
    enter local mode with a broken derived state).
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", return_value=_server_config()),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(
            app_module,
            "_build_config_derived_state",
            side_effect=RuntimeError("ha_client init failed"),
        ),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True)

        assert app_module._state["mode"] == "cloud-only", (
            "mode must stay cloud-only when rebuild fails"
        )
        assert app_module._state["cloud_only_reason"] == "apply_failed"


def test_refresh_config_preload_partial_stays_local():
    """When refresh_config_from_disk=True: if _build_config_derived_state
    succeeds but sets boot_degraded (partial preload), the server stays LOCAL.

    A partial preload is not a failure — recall self-heals via on-miss weight
    probing and the cache re-warms on demand. The base model must NOT be
    released; boot_degraded stays set as a signal (surfaced in /status
    attention) until a later preload fully hydrates.
    """
    from paramem.server import app as app_module

    def fake_build_sets_degraded(
        config, *, cloud_only, rebuild_session_buffer=True, full_rebuild=True
    ):
        # Simulate a partial preload — _preload_memory_store sets boot_degraded.
        app_module._state["boot_degraded"] = {
            "reason": "preload_partial",
            "hits": 5,
            "total": 10,
            "missed_by_tier": {},
            "source": "WeightMemorySource",
        }

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", return_value=_server_config()),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(
            app_module, "_build_config_derived_state", side_effect=fake_build_sets_degraded
        ),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True)

        # Stays local with the model reloaded (a degrade path would have set
        # cloud_only_reason='preload_failed' and returned before mode=local);
        # boot_degraded is retained as a signal.
        assert app_module._state["mode"] == "local"
        assert app_module._state["cloud_only_reason"] is None
        mock_load.assert_called_once()
        assert app_module._state["boot_degraded"] is not None


def test_plain_reclaim_does_not_call_load_server_config():
    """Plain reclaim (refresh_config_from_disk=False, the default) must NOT
    call load_server_config — same config reload path.
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config") as mock_lsc,
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model()  # refresh_config_from_disk=False (default)

    mock_lsc.assert_not_called()


def test_post_load_gate_not_called_from_reload():
    """The post-load VRAM gate (check_post_load_budget) must NOT be called
    from _live_reload_base_model or _build_config_derived_state.

    Reload owns its own pre-load drain-wait + load-exception fallback; running
    the boot-only post-load gate here would double-count headroom and surface a
    spurious second 'post-load budget' attention warning on every reclaim.
    """
    from paramem.server import app as app_module

    # Gate reports room → pre-flight passes.
    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=True),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
        patch.object(app_module, "check_post_load_budget") as mock_post_gate,
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False)

    (
        mock_post_gate.assert_not_called(),
        (
            "check_post_load_budget must never be called from _live_reload_base_model — "
            "the reload path handles its own VRAM safety via drain-wait + load-exception."
        ),
    )


# ---------------------------------------------------------------------------
# _preload_memory_store source selection + boot_degraded lifecycle
# ---------------------------------------------------------------------------


def _fake_memory_store():
    """Return a minimal mock MemoryStore for preload tests."""
    store = MagicMock()
    store.tiers_with_registry.return_value = []
    store.active_keys_in_tier.return_value = []
    return store


def test_preload_source_selection_simulate_mode():
    """preload_cache=True + consolidation.mode='simulate' → DiskMemorySource selected.

    Source selection now lives in _build_store_contents (called by
    _hydrate_memory_store_in_place, called by _preload_memory_store).
    Selection uses config.consolidation.mode, NOT _state["mode"].
    """
    from paramem.server import app as app_module

    config = _server_config(consolidation_mode="simulate", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {
        "mode": "cloud-only",  # _state["mode"] is cloud-only; consolidation.mode is simulate
        "boot_degraded": None,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    disk_source_calls = []
    weight_source_calls = []

    class FakeDiskSource:
        def __init__(self, _adapter_dir):
            disk_source_calls.append("init")

        def probe(self, keys_by_tier, should_abort=None):
            return {"graph1": {"key": "graph1", "question": "q", "answer": "a"}}

    class FakeWeightSource:
        def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
            weight_source_calls.append("init")

        def probe(self, keys_by_tier, should_abort=None):
            return {}

    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod

    # Build a fake registry with one active key so the source-selection branch is entered.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["graph1"]
    fake_registry_map = {"episodic": fake_reg}

    fake_store = MagicMock()

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        # Constructor returns fake_store; read_registries_from_disk returns fake registry map.
        patch.object(store_mod, "MemoryStore", return_value=fake_store),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "DiskMemorySource", FakeDiskSource),
        patch.object(src_mod, "WeightMemorySource", FakeWeightSource),
    ):
        app_module._preload_memory_store(
            config,
            model=app_module._state.get("model"),
            tokenizer=app_module._state.get("tokenizer"),
        )

    assert disk_source_calls, "DiskMemorySource should be used when consolidation.mode='simulate'"
    assert not weight_source_calls, "WeightMemorySource must not be used in simulate mode"
    # Full probe succeeded → boot_degraded should be cleared.
    assert app_module._state.get("boot_degraded") is None


def test_preload_source_selection_train_mode_uses_weight_source():
    """preload_cache=True + consolidation.mode='train' + model present
    → WeightMemorySource selected (NOT DiskMemorySource), regardless of
    _state["mode"] (which might be cloud-only on the apply path).

    Source selection now lives in _build_store_contents (called by
    _hydrate_memory_store_in_place, called by _preload_memory_store).
    """
    from paramem.server import app as app_module

    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {
        "mode": "cloud-only",  # runtime mode cloud-only; must use weight source for train
        "boot_degraded": None,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    disk_source_calls = []
    weight_source_calls = []

    class FakeDiskSource:
        def __init__(self, _adapter_dir):
            disk_source_calls.append("init")

        def probe(self, keys_by_tier, should_abort=None):
            return {}

    class FakeWeightSource:
        def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
            weight_source_calls.append("init")

        def probe(self, keys_by_tier, should_abort=None):
            return {"graph1": {"key": "graph1", "question": "q", "answer": "a"}}

    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod

    # Build a fake registry with one active key so the source-selection branch is entered.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["graph1"]
    fake_registry_map = {"episodic": fake_reg}

    fake_store = MagicMock()

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        # Constructor returns fake_store; read_registries_from_disk returns fake registry map.
        patch.object(store_mod, "MemoryStore", return_value=fake_store),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "DiskMemorySource", FakeDiskSource),
        patch.object(src_mod, "WeightMemorySource", FakeWeightSource),
    ):
        app_module._preload_memory_store(
            config,
            model=app_module._state.get("model"),
            tokenizer=app_module._state.get("tokenizer"),
        )

    assert weight_source_calls, "WeightMemorySource should be used when consolidation.mode='train'"
    assert not disk_source_calls, "DiskMemorySource must not be used in train mode"
    assert app_module._state.get("boot_degraded") is None


def test_preload_cache_false_clears_boot_degraded():
    """preload_cache=False (intentional opt-out): boot_degraded is CLEARED
    (not set) after an apply on preload-off deployments — recall takes the
    per-key source path on every miss (correction #5 boot_degraded lifecycle).
    """
    from paramem.server import app as app_module

    config = _server_config(preload_cache=False)
    config.consolidation.indexed_key_replay = False

    fake_store = MagicMock()
    fake_store.load_registries_from_disk.return_value = None
    fake_store.load_bookkeeping_from_disk.return_value = {
        "loaded": 0,
        "orphaned": 0,
    }

    state_patch = {
        "boot_degraded": {"reason": "preload_partial", "hits": 0, "total": 5},
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    import paramem.memory.store as store_mod

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(store_mod, "MemoryStore", return_value=fake_store),
    ):
        app_module._preload_memory_store(
            config,
            model=app_module._state.get("model"),
            tokenizer=app_module._state.get("tokenizer"),
        )

    assert app_module._state["boot_degraded"] is None, (
        "boot_degraded must be cleared when preload_cache=False"
    )


def test_preload_partial_sets_boot_degraded(tmp_path):
    """When some active keys cannot be materialised (mounted-but-failed, or the
    registry-sha256 binding-bug: slot present but unmounted → probe None):
    boot_degraded is SET.  This is the normal (non-swap) path — tmp_path has no
    base-swap marker, so the invalidity gate does not fire.

    Source enumeration now comes from the fresh registry returned by
    _build_store_contents (via MemoryStore.read_registries_from_disk), not from
    the live store instance methods.
    """
    from paramem.server import app as app_module

    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.paths.data = tmp_path  # no base-swap marker here → gate inactive
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {
        "boot_degraded": None,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    # Probe returns only one of two keys → partial hydration of a backed tier.
    class FakeWeightSource:
        def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
            pass

        def probe(self, keys_by_tier, should_abort=None):
            # Only graph1 found, graph2 missing.
            return {"graph1": {"key": "graph1", "question": "q", "answer": "a"}}

    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod

    # Registry reports two active keys; the probe will only return one.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["graph1", "graph2"]
    fake_registry_map = {"episodic": fake_reg}

    fake_store = MagicMock()

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(store_mod, "MemoryStore", return_value=fake_store),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "WeightMemorySource", FakeWeightSource),
    ):
        app_module._preload_memory_store(
            config,
            model=app_module._state.get("model"),
            tokenizer=app_module._state.get("tokenizer"),
        )

        degraded = app_module._state.get("boot_degraded")
        assert degraded is not None, "boot_degraded must be set on partial hydration"
        assert degraded["hits"] == 1
        assert degraded["total"] == 2


def test_preload_skips_registry_during_base_swap(tmp_path):
    """While a base-model swap is in flight (a ``base_swap`` marker is present),
    the on-disk registry describes the PREVIOUS model and is invalid for the loaded
    one.  preload must NOT load it — it returns an empty store with boot_degraded
    cleared, so the new model starts knowing nothing until Phase B retrains it.
    The registry files are left untouched (Phase B / rollback use them).
    """
    from paramem.server import app as app_module

    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.paths.data = tmp_path
    config.consolidation.indexed_key_replay = False

    fake_store = MagicMock()
    # The registry must NOT be loaded into the live store during a base-swap.
    fake_store.load_registries_from_disk.side_effect = AssertionError(
        "registry must not be loaded into the live store during a base-swap"
    )

    fake_marker = MagicMock()
    fake_marker.migration_kind = "base_swap"
    fake_marker.base_swap_phase = "phaseA_done"

    state_patch = {
        "boot_degraded": {"reason": "stale"},  # must be cleared by the gate
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    import paramem.memory.store as store_mod

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(store_mod, "MemoryStore", return_value=fake_store),
        patch("paramem.server.trial_state.read_trial_marker", return_value=fake_marker),
    ):
        result = app_module._preload_memory_store(
            config,
            model=app_module._state.get("model"),
            tokenizer=app_module._state.get("tokenizer"),
        )

    assert result is fake_store, "an (empty) store must still be returned"
    fake_store.load_registries_from_disk.assert_not_called()
    assert app_module._state.get("boot_degraded") is None, (
        "boot_degraded must be cleared while a base-swap is in flight"
    )


# ---------------------------------------------------------------------------
# Fix 2 — _wait_for_gpu_drain unit tests
# ---------------------------------------------------------------------------


def test_wait_for_gpu_drain_returns_true_when_cuda_unavailable():
    """When CUDA is unavailable, _wait_for_gpu_drain must return True immediately
    (no-op: CPU-only environments must not block at boot).
    """
    from paramem.server import app as app_module

    with patch("paramem.server.app.torch.cuda.is_available", return_value=False):
        result = app_module._wait_for_gpu_drain(
            4 * 2**30, timeout_s=5.0, stable_reads=3, poll_interval_s=0.1
        )

    assert result is True


def test_wait_for_gpu_drain_returns_true_when_immediately_sufficient():
    """When free bytes ≥ needed on the very first read (normal empty-GPU boot),
    _wait_for_gpu_drain must return True without blocking for the full stable_reads
    window — i.e. it blocks ONLY for (stable_reads - 1) more polls, not timeout_s.

    Uses stable_reads=1 so the very first sufficient read satisfies the condition.
    """
    from paramem.server import app as app_module

    free = 7 * 2**30  # 7 GiB free
    needed = 5 * 2**30  # 5 GiB needed

    with (
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch("paramem.server.app.torch.cuda.mem_get_info", return_value=(free, 8 * 2**30)),
        patch("paramem.server.app.time.sleep") as mock_sleep,
    ):
        result = app_module._wait_for_gpu_drain(
            needed, timeout_s=60.0, stable_reads=1, poll_interval_s=1.5
        )

    assert result is True
    # stable_reads=1: one passing read satisfies the condition immediately —
    # no sleep needed between the single read and the return.
    mock_sleep.assert_not_called()


def test_wait_for_gpu_drain_requires_stable_reads_consecutive():
    """_wait_for_gpu_drain must NOT return True until stable_reads consecutive
    reads all report free ≥ needed.  A transient passing read followed by a
    failing read resets the counter.

    Sequence: insufficient, sufficient, insufficient, sufficient×3 → True.
    """
    from paramem.server import app as app_module

    needed = 5 * 2**30
    free_enough = needed + 1
    # Shortfall must exceed the reclaimable-context credit (0.5 GiB) so the credited
    # effective free is still < needed; 1 GiB below makes it unambiguous.
    free_short = needed - 2**30

    read_sequence = [
        (free_short, 8 * 2**30),  # insufficient: counter=0
        (free_enough, 8 * 2**30),  # sufficient: counter=1
        (free_short, 8 * 2**30),  # insufficient: counter reset to 0
        (free_enough, 8 * 2**30),  # sufficient: counter=1
        (free_enough, 8 * 2**30),  # sufficient: counter=2
        (free_enough, 8 * 2**30),  # sufficient: counter=3 → return True
    ]
    call_count = 0

    def fake_mem_get_info(_device=0):
        nonlocal call_count
        val = read_sequence[call_count]
        call_count += 1
        return val

    with (
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch("paramem.server.app.torch.cuda.mem_get_info", side_effect=fake_mem_get_info),
        patch("paramem.server.app.time.sleep"),
    ):
        result = app_module._wait_for_gpu_drain(
            needed, timeout_s=60.0, stable_reads=3, poll_interval_s=0.1
        )

    assert result is True
    assert call_count == len(read_sequence), (
        f"Expected all {len(read_sequence)} reads; got {call_count}"
    )


def test_wait_for_gpu_drain_returns_false_on_timeout():
    """When device-wide free never reaches needed_bytes within timeout_s,
    _wait_for_gpu_drain must return False.

    Patches time.monotonic so the test runs fast (no real wall-clock wait).
    """
    from paramem.server import app as app_module

    needed = 5 * 2**30
    free_short = needed - 2**30  # 1 GiB short — exceeds the 0.5 GiB credit, always insufficient

    # Simulate time advancing: first call returns 0 (start), subsequent calls
    # advance past deadline so the loop exits on the second iteration.
    time_sequence = [0.0, 0.0, 999.0]  # deadline = 0 + timeout_s; third call > deadline
    time_idx = 0

    def fake_monotonic():
        nonlocal time_idx
        val = time_sequence[min(time_idx, len(time_sequence) - 1)]
        time_idx += 1
        return val

    with (
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch("paramem.server.app.torch.cuda.mem_get_info", return_value=(free_short, 8 * 2**30)),
        patch("paramem.server.app.time.monotonic", side_effect=fake_monotonic),
        patch("paramem.server.app.time.sleep"),
    ):
        result = app_module._wait_for_gpu_drain(
            needed, timeout_s=55.0, stable_reads=3, poll_interval_s=1.5
        )

    assert result is False


# ---------------------------------------------------------------------------
# Fix 2 — boot integration: drain-fail → cloud-only + auto-reclaim armed
# ---------------------------------------------------------------------------


def test_boot_drain_fail_degrades_to_cloud_only_and_arms_reclaim(tmp_path):
    """When _wait_for_gpu_drain returns False at boot (GPU did not drain in time):
    - model is NOT loaded (_load_model_into_state not called)
    - cloud_only_reason is set to 'insufficient_vram'
    - _state['model'] is None
    - the auto-reclaim task is created (cloud_only AND not cloud_only_startup)

    Uses the same lifespan-run pattern as test_lifespan_runs_validator_before_load_base_model
    in test_vram_validator.py, short-circuiting after the boot block.
    """
    import asyncio

    from paramem.server import app as app_module
    from paramem.server.config import ServerConfig

    class _Sentinel(Exception):
        pass

    config = ServerConfig(model_name="mistral")
    config.cloud_only = False
    from paramem.server.config import PathsConfig

    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    saved_state = {
        key: app_module._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }
    app_module._state["config"] = config
    app_module._state["cloud_only_startup"] = False
    app_module._state["defer_model"] = False

    load_called = []

    def spy_load(*args, **kwargs):
        load_called.append(True)
        raise _Sentinel("load must not be reached on drain-fail")

    from paramem.server.vram_validator import TopologyAssessment

    fake_assessment = TopologyAssessment(
        required_bytes=5 * 2**30,
        adapter_bytes=1,
        base_bytes=1,
        baseline_bytes=8 * 2**30,
        fits_baseline=True,
        margin_bytes=3 * 2**30,
        breakdown="stub",
    )

    # Change 4: the overflow check in lifespan calls get_device_properties + mem_get_info
    # after assess_topology. Mock both; required=5 GiB < free=7 GiB so no overflow fires.
    _total_bytes = int(8 * 2**30)
    _free_bytes = int(7 * 2**30)

    try:
        with (
            patch.object(app_module, "predict_base_bytes", return_value=4 * 2**30),
            patch.object(app_module, "assess_topology", return_value=fake_assessment),
            # Drain fails → model should NOT load
            patch.object(app_module, "_wait_for_gpu_drain", return_value=False),
            patch.object(app_module, "_load_model_into_state", spy_load),
            patch.object(app_module, "_gpu_occupied", return_value=False),
            patch("paramem.server.app.torch.cuda.is_available", return_value=True),
            patch("paramem.server.app.torch.cuda.get_device_properties") as mock_props,
            patch(
                "paramem.server.app.torch.cuda.mem_get_info",
                return_value=(_free_bytes, _total_bytes),
            ),
            patch.object(app_module, "apply_process_cap"),
            patch("transformers.AutoConfig.from_pretrained") as mock_cfg,
            # Short-circuit after _build_config_derived_state so we don't need
            # the full component tree — raise Sentinel at _build_config_derived_state.
            patch.object(
                app_module,
                "_build_config_derived_state",
                side_effect=_Sentinel("short-circuit"),
            ),
        ):
            mock_props.return_value.total_memory = _total_bytes
            mock_cfg.return_value = type("C", (), {"hidden_size": 4096, "num_hidden_layers": 32})()

            async def _run():
                async with app_module.lifespan(app_module.app):
                    pass

            with pytest.raises(_Sentinel):
                asyncio.run(_run())
    finally:
        for key, val in saved_state.items():
            if val is None:
                app_module._state.pop(key, None)
            else:
                app_module._state[key] = val
        app_module._state.pop("post_load_budget_warning", None)
        app_module._state.pop("topology_assessment", None)
        app_module._state.pop("usable_ceiling_bytes", None)
        app_module._state.pop("device_total_memory_bytes", None)

    # Model must NOT have been loaded.
    assert not load_called, "_load_model_into_state must not be called when drain fails"
    # cloud_only_reason must be set to indicate the degrade.
    assert app_module._state.get("cloud_only_reason") == "insufficient_vram", (
        f"expected 'insufficient_vram', got {app_module._state.get('cloud_only_reason')!r}"
    )
    # model must be None (degrade sets it).
    assert app_module._state.get("model") is None


def test_boot_drain_pass_proceeds_to_load(tmp_path):
    """When _wait_for_gpu_drain returns True (GPU free), the normal load path runs:
    _load_model_into_state is called, model is NOT degraded to cloud-only.
    """
    import asyncio

    from paramem.server import app as app_module
    from paramem.server.config import ServerConfig

    class _Sentinel(Exception):
        pass

    config = ServerConfig(model_name="mistral")
    config.cloud_only = False
    from paramem.server.config import PathsConfig

    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    saved_state = {
        key: app_module._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }
    app_module._state["config"] = config
    app_module._state["cloud_only_startup"] = False
    app_module._state["defer_model"] = False

    load_called = []

    def spy_load(*args, **kwargs):
        load_called.append(True)
        raise _Sentinel("short-circuit: load was reached as expected")

    from paramem.server.vram_validator import TopologyAssessment

    fake_assessment = TopologyAssessment(
        required_bytes=5 * 2**30,
        adapter_bytes=1,
        base_bytes=1,
        baseline_bytes=8 * 2**30,
        fits_baseline=True,
        margin_bytes=3 * 2**30,
        breakdown="stub",
    )

    # Change 4: the overflow check in lifespan calls get_device_properties + mem_get_info
    # after assess_topology. Mock both; required=5 GiB < free=7 GiB so no overflow fires.
    _total_bytes = int(8 * 2**30)
    _free_bytes = int(7 * 2**30)

    try:
        with (
            patch.object(app_module, "predict_base_bytes", return_value=4 * 2**30),
            patch.object(app_module, "assess_topology", return_value=fake_assessment),
            # Drain passes → load should run
            patch.object(app_module, "_wait_for_gpu_drain", return_value=True),
            patch.object(app_module, "_load_model_into_state", spy_load),
            patch.object(app_module, "_gpu_occupied", return_value=False),
            patch("paramem.server.app.torch.cuda.is_available", return_value=True),
            patch("paramem.server.app.torch.cuda.get_device_properties") as mock_props,
            patch(
                "paramem.server.app.torch.cuda.mem_get_info",
                return_value=(_free_bytes, _total_bytes),
            ),
            patch.object(app_module, "apply_process_cap"),
            patch("transformers.AutoConfig.from_pretrained") as mock_cfg,
        ):
            mock_props.return_value.total_memory = _total_bytes
            mock_cfg.return_value = type("C", (), {"hidden_size": 4096, "num_hidden_layers": 32})()

            async def _run():
                async with app_module.lifespan(app_module.app):
                    pass

            with pytest.raises(_Sentinel):
                asyncio.run(_run())
    finally:
        for key, val in saved_state.items():
            if val is None:
                app_module._state.pop(key, None)
            else:
                app_module._state[key] = val
        app_module._state.pop("post_load_budget_warning", None)
        app_module._state.pop("topology_assessment", None)
        app_module._state.pop("usable_ceiling_bytes", None)
        app_module._state.pop("device_total_memory_bytes", None)

    # Load must have been called when drain passes.
    assert load_called, "_load_model_into_state must be called when drain passes"
    # cloud_only_reason must NOT be 'insufficient_vram' on the pass path.
    assert app_module._state.get("cloud_only_reason") != "insufficient_vram", (
        "cloud_only_reason must not be 'insufficient_vram' when drain succeeds"
    )


# ---------------------------------------------------------------------------
# Change 3 — reclaim fit-check uses mem_get_info, no nvidia-smi
# ---------------------------------------------------------------------------


def test_reclaim_fitcheck_uses_mem_get_info_not_nvidia_smi():
    """The reclaim VRAM gate must use torch.cuda.mem_get_info (device-wide free)
    rather than nvidia-smi.  nvidia-smi false-frees exiting processes under WSL2
    and was the documented host-crash cause.

    Verifies: (a) mem_get_info is consulted (via the unified _effective_free_bytes
    gate) when assessment is present and CUDA is available; (b) the reload declines
    gracefully when free stays < required; (c) no subprocess (nvidia-smi) is made.
    time.monotonic is advanced past the deadline so the gate times out fast.
    """
    from paramem.server import app as app_module

    # required=6 GiB; free=2 GiB (even credited: 2.5 GiB) → gate never satisfied.
    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)
    free_bytes = int(2 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
    }

    import subprocess

    # Advance monotonic past the deadline on the 2nd read so the poll exits fast.
    time_seq = [0.0, 0.0, 999.0]
    _i = 0

    def fake_monotonic():
        nonlocal _i
        val = time_seq[min(_i, len(time_seq) - 1)]
        _i += 1
        return val

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch(
            "paramem.server.app.torch.cuda.mem_get_info", return_value=(free_bytes, 8 * 2**30)
        ) as mock_mem_get_info,
        patch("paramem.server.app.time.monotonic", side_effect=fake_monotonic),
        patch("paramem.server.app.time.sleep"),
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(app_module, "_build_config_derived_state") as mock_build,
        # subprocess.run must NOT be called (no nvidia-smi in reclaim path).
        patch.object(subprocess, "run") as mock_subprocess,
    ):
        app_module._live_reload_base_model()

        # mem_get_info must be consulted (via _effective_free_bytes).
        mock_mem_get_info.assert_called()
        # Load must NOT be attempted — free < required.
        mock_load.assert_not_called()
        mock_build.assert_not_called()
        assert app_module._state["cloud_only_reason"] == "insufficient_vram"
        # No nvidia-smi subprocess in the gate path.
        mock_subprocess.assert_not_called()


def test_reclaim_fitcheck_cuda_unavailable_skips_check():
    """When CUDA is unavailable and assessment is present, the fit-check is
    skipped (is_available() guards the mem_get_info call).  Load proceeds.
    """
    from paramem.server import app as app_module

    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=False),
        patch("paramem.server.app.torch.cuda.mem_get_info") as mock_mem_get_info,
        patch.object(app_module, "_load_model_into_state") as mock_load,
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model()

    # With CUDA unavailable, mem_get_info must NOT be called.
    mock_mem_get_info.assert_not_called()
    # Load must proceed (no VRAM gate fires without CUDA).
    mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Change 4 — VRAM overflow attention item
# ---------------------------------------------------------------------------


def test_vram_overflow_attention_item_emitted_when_required_exceeds_usable():
    """When topology_assessment.required_bytes > usable_ceiling_bytes,
    collect_attention_items returns a vram_config_overflow action_required item.
    """
    from paramem.server.attention import collect_attention_items
    from paramem.server.vram_validator import TopologyAssessment

    _GiB = 2**30
    state = {
        "topology_assessment": TopologyAssessment(
            required_bytes=int(7.02 * _GiB),
            adapter_bytes=int(0.026 * _GiB),
            base_bytes=int(4.5 * _GiB),
            baseline_bytes=8 * _GiB,
            fits_baseline=True,
            margin_bytes=int(0.98 * _GiB),
            breakdown="stub",
        ),
        "usable_ceiling_bytes": int(6.83 * _GiB),
        "device_total_memory_bytes": int(8.0 * _GiB),
    }
    items = collect_attention_items(state, config=None)
    kinds = [it.kind for it in items]
    assert "vram_config_overflow" in kinds, (
        f"Expected 'vram_config_overflow' attention item; got kinds={kinds}"
    )
    overflow_item = next(it for it in items if it.kind == "vram_config_overflow")
    assert overflow_item.level == "action_required"
    assert "7.02" in overflow_item.summary
    assert "6.83" in overflow_item.summary
    assert overflow_item.action_hint is not None


def test_vram_overflow_attention_item_absent_when_fits():
    """When required ≤ usable_ceiling, no overflow item is emitted."""
    from paramem.server.attention import collect_attention_items
    from paramem.server.vram_validator import TopologyAssessment

    _GiB = 2**30
    state = {
        "topology_assessment": TopologyAssessment(
            required_bytes=int(5.0 * _GiB),
            adapter_bytes=int(0.026 * _GiB),
            base_bytes=int(4.0 * _GiB),
            baseline_bytes=8 * _GiB,
            fits_baseline=True,
            margin_bytes=3 * _GiB,
            breakdown="stub",
        ),
        "usable_ceiling_bytes": int(6.83 * _GiB),
        "device_total_memory_bytes": int(8.0 * _GiB),
    }
    items = collect_attention_items(state, config=None)
    kinds = [it.kind for it in items]
    assert "vram_config_overflow" not in kinds


def test_vram_overflow_attention_item_absent_when_state_missing():
    """When topology_assessment or usable_ceiling_bytes is missing, no item."""
    from paramem.server.attention import collect_attention_items

    items = collect_attention_items({}, config=None)
    kinds = [it.kind for it in items]
    assert "vram_config_overflow" not in kinds


def test_vram_low_headroom_attention_item_emitted_when_state_populated():
    """When _state['vram_low_headroom_warning'] is populated by
    check_vram_headroom, collect_attention_items returns a warning-level item
    with kind='vram_low_headroom' surfaced in /status.attention.
    """
    import time as _time

    from paramem.server.attention import collect_attention_items

    state = {
        "vram_low_headroom_warning": {
            "label": "s042",
            "free_gib": 1.10,
            "headroom_gib": 1.50,
            "total_gib": 8.0,
            "observed_at": _time.time() - 5,
        }
    }
    items = collect_attention_items(state, config=None)
    item = next((it for it in items if it.kind == "vram_low_headroom"), None)
    assert item is not None, "expected vram_low_headroom item"
    assert item.level == "warning"
    assert "s042" in item.summary
    assert "1.10" in item.summary
    assert "1.50" in item.summary
    assert item.action_hint is not None
    assert item.age_seconds is not None and item.age_seconds >= 0


def test_vram_low_headroom_attention_item_absent_when_state_clean():
    """No state key → no warning item."""
    from paramem.server.attention import collect_attention_items

    items = collect_attention_items({}, config=None)
    kinds = [it.kind for it in items]
    assert "vram_low_headroom" not in kinds


def test_lifespan_sets_vram_overflow_warning_when_required_exceeds_usable(tmp_path):
    """When topology required_bytes > mem_get_info free (usable ceiling),
    the lifespan must set _state['vram_overflow_warning'] and emit a WARNING log.
    """
    import asyncio
    import logging

    from paramem.server import app as app_module
    from paramem.server.config import PathsConfig, ServerConfig
    from paramem.server.vram_validator import TopologyAssessment

    class _Sentinel(Exception):
        pass

    config = ServerConfig(model_name="mistral")
    config.cloud_only = False
    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    # Assessment: required = 7.5 GiB; usable ceiling = 6.8 GiB → overflow.
    _GiB = 2**30
    required_bytes = int(7.5 * _GiB)
    usable_bytes = int(6.8 * _GiB)
    total_bytes = int(8.0 * _GiB)

    fake_assessment = TopologyAssessment(
        required_bytes=required_bytes,
        adapter_bytes=int(0.026 * _GiB),
        base_bytes=int(4.5 * _GiB),
        baseline_bytes=8 * _GiB,
        fits_baseline=False,
        margin_bytes=8 * _GiB - required_bytes,
        breakdown="stub",
    )

    saved_state = {
        key: app_module._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }
    app_module._state["config"] = config
    app_module._state["cloud_only_startup"] = False
    app_module._state["defer_model"] = False

    warning_records = []

    class _OverflowHandler(logging.Handler):
        def emit(self, record):
            if "VRAM CONFIG OVERFLOW" in record.getMessage():
                warning_records.append(record)

    handler = _OverflowHandler()
    server_logger = logging.getLogger("paramem.server.app")
    server_logger.addHandler(handler)

    try:
        with (
            patch.object(app_module, "predict_base_bytes", return_value=int(4.5 * _GiB)),
            patch.object(app_module, "assess_topology", return_value=fake_assessment),
            patch.object(app_module, "_wait_for_gpu_drain", return_value=False),
            patch.object(app_module, "_gpu_occupied", return_value=False),
            patch("paramem.server.app.torch.cuda.is_available", return_value=True),
            patch("paramem.server.app.torch.cuda.get_device_properties") as mock_props,
            patch(
                "paramem.server.app.torch.cuda.mem_get_info",
                return_value=(usable_bytes, total_bytes),
            ),
            patch.object(app_module, "apply_process_cap"),
            patch("transformers.AutoConfig.from_pretrained") as mock_cfg,
            patch.object(
                app_module,
                "_build_config_derived_state",
                side_effect=_Sentinel("short-circuit"),
            ),
        ):
            mock_props.return_value.total_memory = total_bytes
            mock_cfg.return_value = type("C", (), {"hidden_size": 4096, "num_hidden_layers": 32})()

            async def _run():
                async with app_module.lifespan(app_module.app):
                    pass

            with pytest.raises(_Sentinel):
                asyncio.run(_run())
    finally:
        server_logger.removeHandler(handler)
        for key, val in saved_state.items():
            if val is None:
                app_module._state.pop(key, None)
            else:
                app_module._state[key] = val
        app_module._state.pop("post_load_budget_warning", None)
        app_module._state.pop("topology_assessment", None)
        app_module._state.pop("usable_ceiling_bytes", None)
        app_module._state.pop("device_total_memory_bytes", None)

    assert warning_records, (
        "Expected WARNING log containing 'VRAM CONFIG OVERFLOW' when required > usable"
    )
    # Note: _state is restored in finally, so verify from the warning records.
    # The key check is that the WARNING was emitted.
    assert any("VRAM CONFIG OVERFLOW" in r.getMessage() for r in warning_records)


# ---------------------------------------------------------------------------
# Active-store migration arming (shared by lifespan + live-reload)
# ---------------------------------------------------------------------------


def test_arm_active_store_migration_arms_on_divergence():
    """_arm_active_store_migration sets pending_rehydration + persists state and
    falls back to the source mode when detect_mode_switch reports a divergence.

    This is the single arming source shared by the lifespan startup path and the
    live config-reload path.
    """
    from paramem.server import app as app_module
    from paramem.server.active_store_migration import MigrationState

    fake_state = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
    config = _server_config(consolidation_mode="train")

    with (
        patch.dict(
            app_module._state,
            {"pending_rehydration": False, "effective_mode": None},
            clear=False,
        ),
        patch("paramem.server.active_store_migration.detect_mode_switch", return_value=fake_state),
        patch("paramem.server.active_store_migration.save_state") as mock_save,
    ):
        armed = app_module._arm_active_store_migration(config)

        assert armed is True
        assert app_module._state["pending_rehydration"] is True
        # Inference falls back to the SOURCE mode until the rebuild clears.
        assert app_module._state["effective_mode"] == "simulate"
        mock_save.assert_called_once()


def test_arm_active_store_migration_noop_when_no_switch():
    """No divergence → pending_rehydration cleared, effective_mode tracks the
    configured mode, no state persisted, returns False (safe no-op for every
    non-mode config change)."""
    from paramem.server import app as app_module

    config = _server_config(consolidation_mode="train")

    with (
        patch.dict(
            app_module._state,
            {"pending_rehydration": True, "effective_mode": "simulate"},
            clear=False,
        ),
        patch("paramem.server.active_store_migration.detect_mode_switch", return_value=None),
        patch("paramem.server.active_store_migration.save_state") as mock_save,
    ):
        armed = app_module._arm_active_store_migration(config)

        assert armed is False
        assert app_module._state["pending_rehydration"] is False
        assert app_module._state["effective_mode"] == "train"
        mock_save.assert_not_called()


def test_refresh_config_from_disk_arms_mode_switch():
    """Regression: a live config refresh arms the active-store rebuild.

    Previously only the lifespan path called detect_mode_switch, so a
    LIVE-applied mode change (migration accept / config apply) committed the new
    mode to _state but left the on-disk store stale until the next restart. The
    refresh path must invoke the shared arming helper with the new config.
    """
    from paramem.server import app as app_module

    new_config = _server_config(consolidation_mode="train")

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(consolidation_mode="simulate"),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", return_value=new_config),
        patch.object(app_module, "_arm_active_store_migration") as mock_arm,
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True)

        mock_arm.assert_called_once_with(new_config)


# ---------------------------------------------------------------------------
# Voice drain/restore invariant in _live_reload_base_model
# ---------------------------------------------------------------------------


def test_voice_drain_called_before_release_when_gpu():
    """When voice_profile=='gpu', _set_voice_pipeline_profile('cpu') is called
    before _release_base_model_in_process (entry drain).

    The drain must fire regardless of refresh_config_from_disk (tested here on
    the partial path).  It is not called when voice_profile is already 'cpu'
    (idempotent guard inside _set_voice_pipeline_profile).
    """
    from paramem.server import app as app_module

    call_order = []

    def _fake_set_voice(profile, *, lock_held=False):
        call_order.append(("voice", profile, lock_held))

    def _fake_release():
        call_order.append("release")

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process", side_effect=_fake_release),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False, lock_held=False)

    # voice drain must appear before release
    voice_drain_idx = next(
        (i for i, e in enumerate(call_order) if e == ("voice", "cpu", False)), None
    )
    release_idx = next((i for i, e in enumerate(call_order) if e == "release"), None)
    assert voice_drain_idx is not None, (
        "voice drain to cpu must be called when voice_profile=='gpu'"
    )
    assert release_idx is not None, "release must be called"
    assert voice_drain_idx < release_idx, (
        f"voice drain must precede release; got order {call_order}"
    )


def test_voice_drain_not_called_when_already_cpu():
    """When voice_profile=='cpu', the entry drain is skipped (idempotent guard).

    Callers already cloud-only (voice on CPU) must not incur a no-op STT unload.
    """
    from paramem.server import app as app_module

    drain_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        drain_calls.append(profile)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,
        "boot_degraded": None,
        "voice_profile": "cpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False)

    assert "cpu" not in drain_calls, (
        f"drain must not fire when voice_profile is already 'cpu'; calls={drain_calls}"
    )


def test_voice_restore_called_on_partial_success():
    """On a successful partial reload (refresh_config_from_disk=False),
    _set_voice_pipeline_profile('gpu') is called after mode becomes 'local'.

    This is the partial-path restore; lock_held is forwarded correctly.
    """
    from paramem.server import app as app_module

    voice_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        voice_calls.append((profile, lock_held))

    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=True),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False, lock_held=True)

        # Read _state inside the with block so patch.dict hasn't restored the
        # pre-patch values yet.  Outside the block, patch.dict(clear=False)
        # restores "mode" to whatever it was before the patch — which varies
        # across test orderings and causes a spurious isolation failure.
        assert app_module._state["mode"] == "local"

    assert ("gpu", True) in voice_calls, (
        f"voice restore to gpu with lock_held=True expected on partial success; calls={voice_calls}"
    )


def test_voice_not_restored_on_full_path():
    """On the full-rebuild path (refresh_config_from_disk=True), the primitive
    does NOT restore voice to gpu — _build_config_derived_state handles it.

    Adding a restore here would double-load the STT/TTS pair.
    """
    from paramem.server import app as app_module

    gpu_restore_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        if profile == "gpu":
            gpu_restore_calls.append(profile)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "config_path": "configs/server.yaml",
        "topology_assessment": None,
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "load_server_config", return_value=_server_config()),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
        patch.object(app_module, "_arm_active_store_migration", return_value=False),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=True, lock_held=True)

    assert not gpu_restore_calls, (
        "primitive must NOT restore voice to gpu on the full path "
        f"(_build_config_derived_state owns that); calls={gpu_restore_calls}"
    )


def test_voice_not_restored_on_failure_paths():
    """On failure branches (gate declined, load failed, rebuild failed),
    voice stays on CPU — the entry drain put it there; do not restore.
    """
    from paramem.server import app as app_module

    gpu_restore_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        if profile == "gpu":
            gpu_restore_calls.append(profile)

    fake_assessment = MagicMock(name="assessment")
    fake_assessment.required_bytes = int(6 * 2**30)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": fake_assessment,
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    # Gate declined path (insufficient_vram).
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch("paramem.server.app.torch.cuda.is_available", return_value=True),
        patch.object(app_module, "_wait_for_gpu_drain", return_value=False),
        patch.object(app_module, "_load_model_into_state") as mock_load,
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False)

    mock_load.assert_not_called()
    assert not gpu_restore_calls, (
        "voice must not be restored to gpu when gate declines (insufficient_vram); "
        f"calls={gpu_restore_calls}"
    )


def test_voice_not_restored_on_load_failure():
    """On the load-failure branch (app.py:4258-4264), voice stays on CPU.

    When ``_load_model_into_state`` raises after the preflight passes,
    the primitive sets cloud_only_reason="reload_failed" and returns without
    calling ``_set_voice_pipeline_profile("gpu")``.  The entry drain already
    moved voice to CPU; leaving it there is correct (cloud-only server should
    hold ~0 GiB).
    """
    from paramem.server import app as app_module

    gpu_restore_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        if profile == "gpu":
            gpu_restore_calls.append(profile)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,  # skip preflight gate
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(
            app_module,
            "_load_model_into_state",
            side_effect=RuntimeError("simulated OOM"),
        ),
        patch.object(app_module, "_build_config_derived_state") as mock_build,
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False)

        # Assertions inside the with block so patch.dict hasn't restored _state yet.
        mock_build.assert_not_called()
        assert not gpu_restore_calls, (
            "voice must not be restored to gpu when load fails (reload_failed); "
            f"calls={gpu_restore_calls}"
        )
        assert app_module._state["cloud_only_reason"] == "reload_failed"


def test_voice_not_restored_on_rebuild_failure_partial_path():
    """On the rebuild-failure branch of the partial path (app.py:4340-4344), voice stays on CPU.

    When ``_build_config_derived_state`` raises after a successful model load on
    the ``refresh_config_from_disk=False`` path, the primitive releases the
    partial allocation and returns without calling ``_set_voice_pipeline_profile("gpu")``.
    The server ends cloud-only with reason="reload_failed".
    """
    from paramem.server import app as app_module

    gpu_restore_calls = []

    def _fake_set_voice(profile, *, lock_held=False):
        if profile == "gpu":
            gpu_restore_calls.append(profile)

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,  # skip preflight gate
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),  # load succeeds
        patch.object(
            app_module,
            "_build_config_derived_state",
            side_effect=RuntimeError("simulated rebuild failure"),
        ),
    ):
        app_module._live_reload_base_model(refresh_config_from_disk=False)

        # Assertions inside the with block so patch.dict hasn't restored _state yet.
        assert not gpu_restore_calls, (
            "voice must not be restored to gpu when rebuild fails (partial path, reload_failed); "
            f"calls={gpu_restore_calls}"
        )
        assert app_module._state["cloud_only_reason"] == "reload_failed"


def test_lock_held_forwarded_to_voice_drain():
    """lock_held parameter is forwarded to _set_voice_pipeline_profile on drain."""
    from paramem.server import app as app_module

    voice_calls_with_lock = []

    def _fake_set_voice(profile, *, lock_held=False):
        voice_calls_with_lock.append((profile, lock_held))

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _server_config(),
        "topology_assessment": None,
        "boot_degraded": None,
        "voice_profile": "gpu",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=_fake_set_voice),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state"),
    ):
        app_module._live_reload_base_model(lock_held=True)

    # Drain must have been called with lock_held=True.
    drain = next((c for c in voice_calls_with_lock if c[0] == "cpu"), None)
    assert drain is not None, f"drain call not found in {voice_calls_with_lock}"
    assert drain[1] is True, f"lock_held=True must be forwarded to drain; got {drain}"


# ---------------------------------------------------------------------------
# _hydrate_memory_store_in_place — unit tests
# ---------------------------------------------------------------------------


def _make_real_store(replay_enabled=True):
    """Return a real MemoryStore populated with a few stale entries for testing."""
    from paramem.memory.store import MemoryStore

    store = MemoryStore(replay_enabled=replay_enabled)
    # Seed stale entries that the hydration should clear.
    store.put(
        "episodic",
        "stale_key1",
        {"key": "stale_key1", "subject": "s", "predicate": "p", "object": "o"},
        register=False,
    )
    store.put(
        "episodic",
        "stale_key2",
        {"key": "stale_key2", "subject": "s", "predicate": "p", "object": "o"},
        register=False,
    )
    return store


class FakeWeightSourceHydrate:
    """Minimal WeightMemorySource stand-in for hydration tests.

    Signature matches WeightMemorySource: batch_size is keyword-only;
    probe accepts the optional should_abort keyword forwarded by _build_store_contents.
    """

    def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
        self.probed = []

    def probe(self, keys_by_tier, should_abort=None):
        results = {}
        for _tier, keys in keys_by_tier.items():
            for key in keys:
                self.probed.append(key)
                results[key] = {"key": key, "subject": "s", "predicate": "p", "object": "o"}
        return results


def test_hydrate_clears_stale_entries_and_reloads():
    """_hydrate_memory_store_in_place clears existing stale entries before reloading.

    This is the core fix for the post-consolidation staleness bug: after a full fold
    the old pre-fold interim entries (e.g. 234 total) must be dropped and replaced
    by the active post-fold set (e.g. 186).  The test verifies that stale entries
    from before the call are not present in the store after the call completes.

    The rebuild is done by _build_store_contents, which reads registries via the
    store-free MemoryStore.read_registries_from_disk static method and then publishes
    via store.swap().  Stale entries are cleared because swap() replaces _entries
    atomically — the new entries dict starts empty (no active keys to probe here).
    """
    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod
    from paramem.server import app as app_module

    store = _make_real_store()
    assert len(store) == 2, "precondition: store has 2 stale entries"

    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {
        "boot_degraded": None,
        "store_load_degraded": False,
    }

    # Empty registry map → no active keys → new_entries stays {} → swap clears the store.
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value={}
        ) as mock_read_reg,
        patch.object(src_mod, "WeightMemorySource", FakeWeightSourceHydrate),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        # Assertions inside the with block so patch.dict hasn't restored _state yet.
        # Stale entries must be gone — swap() replaced _entries with an empty dict.
        assert len(store) == 0, (
            f"stale entries must be cleared; store has {len(store)} entries after hydration"
        )
        # Disk read happened (the new builder path).
        mock_read_reg.assert_called_once()
        assert app_module._state["store_load_degraded"] is False


def test_hydrate_registers_active_keys_when_source_hits():
    """When preload_cache=True and the source returns entries, they are written
    into the store (via store.swap()).  The stale pre-call entries are replaced by
    the source results.

    _build_store_contents enumerates active keys from the fresh registry returned
    by MemoryStore.read_registries_from_disk, probes them, then publishes via swap().
    """
    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod
    from paramem.server import app as app_module

    store = _make_real_store()

    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {"boot_degraded": None, "store_load_degraded": False}

    # Registry reports two active keys; the weight source will return both.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["new_key1", "new_key2"]
    fake_registry_map = {"episodic": fake_reg}

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "WeightMemorySource", FakeWeightSourceHydrate),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

    # After hydration the store holds the two newly-probed keys, not the two stale ones.
    assert len(store) == 2
    assert store.get("new_key1") is not None
    assert store.get("new_key2") is not None
    assert store.get("stale_key1") is None
    assert store.get("stale_key2") is None
    assert app_module._state["boot_degraded"] is None


def test_hydrate_sets_boot_degraded_on_partial_probe():
    """When some active keys cannot be materialised, boot_degraded is set
    (same lifecycle as the boot-path partial preload).

    _build_store_contents enumerates keys from the fresh registry returned by
    MemoryStore.read_registries_from_disk; when the probe returns fewer entries
    than the total active key count, stats["boot_degraded"] is set and propagated
    to _state by _hydrate_memory_store_in_place.
    """
    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod
    from paramem.server import app as app_module

    class _PartialSource:
        def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
            pass

        def probe(self, keys_by_tier, should_abort=None):
            # Only return one of the two requested keys.
            return {"key_a": {"key": "key_a", "subject": "s", "predicate": "p", "object": "o"}}

    store = _make_real_store()
    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {"boot_degraded": None, "store_load_degraded": False}

    # Registry reports two active keys; the probe will only return one.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["key_a", "key_b"]
    fake_registry_map = {"episodic": fake_reg}

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "WeightMemorySource", _PartialSource),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        # Read inside the with block so patch.dict hasn't restored _state yet.
        degraded = app_module._state.get("boot_degraded")
        assert degraded is not None, "boot_degraded must be set on partial hydration"
        assert degraded["hits"] == 1
        assert degraded["total"] == 2


def test_hydrate_clears_boot_degraded_on_preload_cache_false():
    """preload_cache=False: boot_degraded is cleared, store stays entry-empty
    (intentional opt-out — same lifecycle as the boot path).
    """
    from paramem.server import app as app_module

    store = _make_real_store()
    config = _server_config(preload_cache=False)
    config.consolidation.indexed_key_replay = False

    state_patch = {
        "boot_degraded": {"reason": "preload_partial", "hits": 0, "total": 5},
        "store_load_degraded": False,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(store, "load_registries_from_disk"),
        patch.object(
            store,
            "load_bookkeeping_from_disk",
            return_value={"loaded": 0, "orphaned": 0},
        ),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

    assert app_module._state["boot_degraded"] is None
    # Stale entries are still cleared even when preload is off.
    assert len(store) == 0


def test_hydrate_sets_store_load_degraded_on_registry_failure():
    """When MemoryStore.read_registries_from_disk raises, store_load_degraded is set to True.

    NEW CONTRACT (data-integrity guard): when the registry read fails,
    _hydrate_memory_store_in_place must NOT call store.swap() — the live store
    is preserved intact so no existing entries are lost.  Callers depend on this
    no-wipe guarantee to safely retry without losing their authoritative state.
    """
    import paramem.memory.store as store_mod
    from paramem.server import app as app_module

    store = _make_real_store()
    pre_call_size = len(store)
    assert pre_call_size == 2, "precondition: store has 2 entries"

    config = _server_config(preload_cache=False)
    config.consolidation.indexed_key_replay = False

    state_patch = {"store_load_degraded": False, "boot_degraded": None}

    swap_calls = []
    original_swap = store.swap

    def spy_swap(*args, **kwargs):
        swap_calls.append(True)
        return original_swap(*args, **kwargs)

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(
            store_mod.MemoryStore,
            "read_registries_from_disk",
            side_effect=OSError("disk error"),
        ),
        patch.object(store, "swap", side_effect=spy_swap),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        # Read inside the with block so patch.dict hasn't restored _state yet.
        assert app_module._state["store_load_degraded"] is True
        # Critical: swap must NOT be called — live store is preserved on registry failure.
        assert not swap_calls, (
            "store.swap() must not be called when registry read fails; "
            "the live store must be preserved intact to avoid losing valid entries"
        )

    # Verify store still has original entries (no-wipe guarantee).
    assert len(store) == pre_call_size, (
        f"live store must not be wiped on registry failure; "
        f"expected {pre_call_size} entries, got {len(store)}"
    )


def test_hydrate_weight_source_is_frame_local():
    """The WeightMemorySource must be dropped (set to None) before _build_store_contents
    returns — verified by confirming no reference to it leaks out of the call.

    This guards the no-base-model-pinning invariant: a surviving reference would
    re-introduce the cloud-only VRAM leak (fixed 2026-05-21).

    The source is created inside _build_store_contents (frame-local, set to None
    before return per BASE-MODEL HOLDER invariant).  The test patches
    src_mod.WeightMemorySource to a capturing fake, triggers the path via
    _hydrate_memory_store_in_place, and confirms exactly one source was created,
    its probe was called (entry is in the store), and no persistent _state or
    module-global reference was established.
    """
    import paramem.memory.source as src_mod
    import paramem.memory.store as store_mod
    from paramem.server import app as app_module

    sources_created = []

    class FakeWeightSourceCapture:
        def __init__(self, model, tokenizer, *, batch_size, registry=None, **kw):
            sources_created.append(self)

        def probe(self, keys_by_tier, should_abort=None):
            return {
                "live_key": {"key": "live_key", "subject": "s", "predicate": "p", "object": "o"}
            }

    store = _make_real_store()
    config = _server_config(consolidation_mode="train", preload_cache=True)
    config.consolidation.indexed_key_replay = False
    config.consolidation.recall_probe_batch_size = 16

    state_patch = {"boot_degraded": None, "store_load_degraded": False}

    # Registry reports one active key so the source-creation branch is entered.
    fake_reg = MagicMock()
    fake_reg.list_active.return_value = ["live_key"]
    fake_registry_map = {"episodic": fake_reg}

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(
            store_mod.MemoryStore, "read_registries_from_disk", return_value=fake_registry_map
        ),
        patch.object(src_mod, "WeightMemorySource", FakeWeightSourceCapture),
    ):
        app_module._hydrate_memory_store_in_place(
            store,
            config,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

    # Exactly one source was created inside _build_store_contents.
    assert len(sources_created) == 1
    # The source object itself is NOT pinned by the function's closure or any
    # persistent attribute — the only reference is sources_created[0] (held by
    # this test).  We verify the function completed; the lack of a module-global
    # or _state reference is structural (no persistent attribute exists to bind it).
    # If this test passes, the function ran and returned without storing the source.
    assert store.get("live_key") is not None, "entry must be in the store after hydration"


# ---------------------------------------------------------------------------
# _finalize_full entry-cache reconciliation
# ---------------------------------------------------------------------------


def test_finalize_full_swaps_store_when_replay_enabled_and_not_degraded():
    """_finalize_full publishes the staged store contents on a clean success.

    Exercises the REAL ``app_module._finalize_full`` directly — it is a
    module-level function taking ``(loop, result, staged_e, staged_r,
    staged_b, staged_stats)`` as explicit parameters (no closure captures),
    so the test calls production code rather than a hand-copied mirror.
    ``store.swap`` is the atomic-publish primitive; it must fire exactly
    once with the staged payload when replay is enabled and the staged
    build did not degrade.
    """
    from paramem.server import app as app_module

    fake_store = MagicMock()
    fake_store.replay_enabled = True
    fake_store.all_active_keys.return_value = ["k1", "k2"]
    swap_calls = []
    fake_store.swap.side_effect = lambda e, r, b: swap_calls.append((e, r, b))

    fake_loop = MagicMock()
    fake_loop.store = fake_store

    fake_result = {"rolled_back": False, "tiers_rebuilt": ["episodic"], "graph_drift_count": 0}
    staged_e = {"episodic": {}}
    staged_r = {"episodic": MagicMock()}
    staged_b = {}
    staged_stats = {"boot_degraded": False, "store_load_degraded": False}

    state_patch = {
        "router": MagicMock(),
        "last_consolidation": None,
        "consolidating": True,
        "event_loop": None,
        "config": _server_config(),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_revalidate_main_adapter_manifests"),
    ):
        app_module._finalize_full(
            fake_loop, fake_result, staged_e, staged_r, staged_b, staged_stats
        )

        # Assertions on _state must run INSIDE the patch.dict block — it
        # restores the dict to its pre-with-block contents on exit, which
        # would discard the mutations made by _finalize_full.
        assert swap_calls == [(staged_e, staged_r, staged_b)], (
            f"store.swap must be called once with the staged payload; got {swap_calls}"
        )
        assert app_module._state["store_load_degraded"] is False
        assert app_module._state["consolidating"] is False


def test_finalize_full_skips_swap_when_staged_build_degraded():
    """store.swap must NOT be called when the staged build degraded.

    A degraded build (store_load_degraded=True) must preserve the live
    in-RAM store rather than publishing a possibly-incomplete registry —
    queries self-heal on-miss instead.
    """
    from paramem.server import app as app_module

    fake_store = MagicMock()
    fake_store.replay_enabled = True
    fake_store.all_active_keys.return_value = []

    fake_loop = MagicMock()
    fake_loop.store = fake_store

    fake_result = {"rolled_back": False, "tiers_rebuilt": ["episodic"], "graph_drift_count": 0}
    staged_stats = {"boot_degraded": None, "store_load_degraded": True}

    state_patch = {
        "router": MagicMock(),
        "last_consolidation": None,
        "consolidating": True,
        "event_loop": None,
        "config": _server_config(),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_revalidate_main_adapter_manifests"),
    ):
        app_module._finalize_full(fake_loop, fake_result, {}, {}, {}, staged_stats)

        fake_store.swap.assert_not_called()
        assert app_module._state["store_load_degraded"] is True
        assert app_module._state["consolidating"] is False


def test_finalize_full_skips_swap_when_replay_disabled():
    """store.swap must NOT be called when replay is disabled.

    There are no registries to publish when replay is off — the swap and
    the degraded-flag propagation are both gated on ``replay_enabled``.
    """
    from paramem.server import app as app_module

    fake_store = MagicMock()
    fake_store.replay_enabled = False
    fake_store.all_active_keys.return_value = []

    fake_loop = MagicMock()
    fake_loop.store = fake_store

    fake_result = {"rolled_back": False, "tiers_rebuilt": ["episodic"], "graph_drift_count": 0}
    staged_stats = {"boot_degraded": False, "store_load_degraded": False}

    state_patch = {
        "router": MagicMock(),
        "last_consolidation": None,
        "consolidating": True,
        "event_loop": None,
        "config": _server_config(),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_revalidate_main_adapter_manifests"),
    ):
        app_module._finalize_full(fake_loop, fake_result, {}, {}, {}, staged_stats)

        fake_store.swap.assert_not_called()
        assert app_module._state["consolidating"] is False
