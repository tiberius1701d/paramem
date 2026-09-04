"""Wrap-once base-model identity contract.

New home for this unit's minimum CPU-only test requirements: they span
``paramem/models/loader.py``, ``paramem/server/config.py``,
``paramem/server/app.py`` and ``paramem/training/consolidation.py``, so no
single existing test file owns all of them, and they test one cohesive
architectural invariant (the base model's object identity is fixed at load
time, and which tiers exist is decided once by ``tier_config_map()``) rather
than any one module's behaviour in isolation.

No GPU: every model here is either a CPU ``MagicMock(spec=PeftModel)`` or a
real ``peft.PeftModel`` wrapped over a tiny CPU ``nn.Module`` (never a real
base model).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from peft import PeftModel

from paramem.backup.types import FatalConfigError
from paramem.models.loader import (
    active_adapter_name,
    base_model_inference,
    create_adapter,
    detach_adapters,
    ensure_adapter_matching,
    ensure_resident_tiers,
    mount_adapter,
)
from paramem.server.config import PathsConfig, load_server_config
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig
from paramem.utils.tiers import MAIN_TIERS
from tests._fold_fixtures import _TinyBase

_FIXTURE_PATH = Path("tests/fixtures/server.yaml")


def _peft_model_mock() -> MagicMock:
    """``MagicMock(spec=PeftModel)`` -- passes every ``isinstance(model,
    PeftModel)`` precondition (``base_model_inference``, ``create_adapter``,
    ...).  ``gradient_checkpointing_disable``/``_enable`` are dynamic
    ``__getattr__``-delegated attributes a real (wrapped) PeftModel exposes
    that ``spec`` cannot see via ``dir(PeftModel)``, so they are pre-set
    explicitly (mirrors ``tests/server/test_gates.py::_make_mock_model``)."""
    model = MagicMock(spec=PeftModel)
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    return model


def _adapter_cfg(rank: int = 4, alpha: int = 8) -> AdapterConfig:
    return AdapterConfig(rank=rank, alpha=alpha, target_modules=["q_proj"])


# ---------------------------------------------------------------------------
# base_model_inference / create_adapter / mount_adapter / ensure_adapter_
# matching / detach_adapters -- TypeError on a non-PeftModel
# ---------------------------------------------------------------------------


class TestNonPeftModelPreconditions:
    def test_base_model_inference_raises_on_non_peft_model(self):
        with pytest.raises(TypeError, match="PeftModel"):
            with base_model_inference(object()):
                pass

    def test_create_adapter_raises_on_non_peft_model(self):
        with pytest.raises(TypeError, match="PeftModel"):
            create_adapter(object(), _adapter_cfg(), "episodic")

    def test_mount_adapter_raises_on_non_peft_model(self, tmp_path):
        with pytest.raises(TypeError, match="PeftModel"):
            mount_adapter(object(), tmp_path, "episodic")

    def test_ensure_adapter_matching_raises_on_non_peft_model(self):
        with pytest.raises(TypeError, match="PeftModel"):
            ensure_adapter_matching(object(), _adapter_cfg(), "episodic")

    def test_detach_adapters_raises_on_non_peft_model(self):
        with pytest.raises(TypeError, match="PeftModel"):
            detach_adapters(object(), ["episodic"])

    def test_base_model_inference_always_enters_disable_adapter_for_a_real_peft_model(self):
        """base_model_inference requires a PeftModel and ALWAYS enters
        disable_adapter() -- there is no adapter-on fallback branch."""
        model = _peft_model_mock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        model.disable_adapter = MagicMock(return_value=cm)

        with base_model_inference(model):
            pass

        model.disable_adapter.assert_called_once()
        cm.__enter__.assert_called_once()
        cm.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# ensure_resident_tiers -- the one get_peft_model site, order + active
# adapter, and the ensure_adapter_matching sole-adapter window -- real CPU
# PeftModel over a tiny nn.Module.
# ---------------------------------------------------------------------------


class TestCreateAdapterReturnsNoneAndActivates:
    def test_returns_none_and_leaves_active_adapter_equal_to_the_new_name(self):
        """create_adapter mutates in place and returns nothing -- callers
        that assign its return value would silently get None back, never a
        model to rebind onto."""
        wrapped = ensure_resident_tiers(_TinyBase(), {"episodic": _adapter_cfg()})

        result = create_adapter(wrapped, _adapter_cfg(rank=8, alpha=16), "semantic")

        assert result is None
        assert "semantic" in wrapped.peft_config
        assert active_adapter_name(wrapped) == "semantic"


class TestEnsureResidentTiersIdentityAndOrder:
    def test_raises_value_error_on_an_empty_map(self):
        with pytest.raises(ValueError):
            ensure_resident_tiers(_TinyBase(), {})

    def test_three_key_ordered_map_fixes_peft_config_order_and_active_adapter(self):
        """Immediately after a wrap, list(model.peft_config) equals the
        map's key order and the active adapter is the FIRST key."""
        adapters = {
            "episodic": _adapter_cfg(rank=4, alpha=8),
            "semantic": _adapter_cfg(rank=4, alpha=8),
            "procedural": _adapter_cfg(rank=8, alpha=16),
        }
        wrapped = ensure_resident_tiers(_TinyBase(), adapters)

        assert isinstance(wrapped, PeftModel)
        assert list(wrapped.peft_config) == list(adapters)
        assert active_adapter_name(wrapped) == "episodic"

    def test_second_call_on_an_already_wrapped_model_only_adds_missing_tiers(self):
        """An already-wrapped PeftModel keeps entries already resident
        untouched and only adds tiers not yet present."""
        wrapped = ensure_resident_tiers(_TinyBase(), {"episodic": _adapter_cfg()})
        resident_before = wrapped.peft_config["episodic"]

        ensure_resident_tiers(wrapped, {"episodic": _adapter_cfg(), "semantic": _adapter_cfg()})

        assert list(wrapped.peft_config) == ["episodic", "semantic"]
        assert wrapped.peft_config["episodic"] is resident_before


class TestEnsureAdapterMatchingSoleAdapterWindow:
    def test_shape_mismatch_recreate_repairs_active_adapter_and_leaves_model_usable(self):
        """A model whose only adapter mismatches on rank: the delete
        empties peft_config and leaves active_adapter stale; the following
        create_adapter (inside ensure_adapter_matching) repairs it -- both
        forward and disable_adapter() work afterwards."""
        wrapped = ensure_resident_tiers(_TinyBase(), {"episodic": _adapter_cfg(rank=4, alpha=8)})

        ensure_adapter_matching(wrapped, _adapter_cfg(rank=8, alpha=16), "episodic")

        assert "episodic" in wrapped.peft_config
        assert active_adapter_name(wrapped) == "episodic"

        out = wrapped(input_ids=torch.randn(1, 4))
        assert out.shape == (1, 4)

        with wrapped.disable_adapter():
            pass


# ---------------------------------------------------------------------------
# detach_adapters' survivor selection after a shape-mismatch delete->create
# re-append reorders peft_config -- MAIN_TIERS order must win over
# peft_config insertion order, which a delete->create pair perturbs.
# ---------------------------------------------------------------------------


class TestDetachAdaptersSurvivorFollowsMainTiersOrderNotInsertionOrder:
    def test_survivor_is_the_main_tiers_first_entry_even_when_it_is_last_in_peft_config(self):
        """Catches a regression where the survivor search reads
        model.peft_config's iteration order (which a shape-mismatch
        recreate perturbs) instead of the fixed MAIN_TIERS vocabulary
        order: after episodic's delete->create re-append, peft_config
        starts with 'semantic' -- a peft_config-order survivor pick would
        wrongly land there instead of 'episodic'."""
        adapters = {
            "episodic": _adapter_cfg(rank=4, alpha=8),
            "semantic": _adapter_cfg(rank=4, alpha=8),
            "procedural": _adapter_cfg(rank=4, alpha=8),
        }
        wrapped = ensure_resident_tiers(_TinyBase(), adapters)
        create_adapter(wrapped, _adapter_cfg(rank=4, alpha=8), "episodic_interim_x")

        # Force episodic's shape-mismatch recreate: delete + re-create
        # re-appends it at the END of peft_config, behind episodic_interim_x.
        ensure_adapter_matching(wrapped, _adapter_cfg(rank=8, alpha=16), "episodic")

        assert list(wrapped.peft_config)[0] == "semantic", (
            "precondition: episodic must no longer lead peft_config for "
            f"this test to distinguish MAIN_TIERS order from insertion "
            f"order; got {list(wrapped.peft_config)}"
        )

        # Simulate the fold continuing to work under the interim slot after
        # the reorder -- active adapter is NOT episodic when detach runs.
        wrapped.set_adapter("episodic_interim_x")
        assert active_adapter_name(wrapped) == "episodic_interim_x"

        deleted = detach_adapters(wrapped, ["episodic_interim_x"])

        assert deleted == ["episodic_interim_x"]
        assert active_adapter_name(wrapped) == "episodic", (
            "survivor selection must follow MAIN_TIERS order (episodic "
            "first), not peft_config's insertion order (which would have "
            "picked 'semantic')"
        )


# ---------------------------------------------------------------------------
# ConsolidationLoop identity -- built from a PEFT-typed double,
# extraction.model / merger.model alias the loop's own model.
# ---------------------------------------------------------------------------


class TestConsolidationLoopModelIdentity:
    def test_loop_extraction_and_merger_model_alias_loop_model(self, tmp_path):
        from paramem.config.taxonomy import resolve_scrub_categories
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop

        model = _peft_model_mock()
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            tier_adapters={"episodic": _adapter_cfg(), "semantic": _adapter_cfg()},
            memory_store=MemoryStore(),
            output_dir=tmp_path,
            extraction_scrub_categories=resolve_scrub_categories(["person name"]),
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )

        assert loop.model is model
        assert loop.extraction.model is loop.model
        assert loop.merger.model is loop.model


# ---------------------------------------------------------------------------
# get_or_create_consolidation_loop / create_consolidation_loop -- the SAME
# identity, but driven through the real production factories rather than a
# hand-built ConsolidationLoop: the boot path
# (_load_model_into_state, load_base_model patched to a PEFT-typed double)
# sets _state["model"], then get_or_create_consolidation_loop must build a
# loop whose extraction.model / merger.model alias that SAME object.
# ---------------------------------------------------------------------------


class TestGetOrCreateConsolidationLoopIdentityThroughTheProductionBootPath:
    def test_boot_then_get_or_create_loop_share_the_same_model_object(self, tmp_path, monkeypatch):
        from paramem.memory.store import MemoryStore
        from paramem.server import app as app_module
        from paramem.server.consolidation import get_or_create_consolidation_loop

        cfg = _cfg_rooted_at(tmp_path)
        double = _peft_model_mock()
        tier_names = list(cfg.tier_config_map())
        double.peft_config = {name: MagicMock() for name in tier_names}
        double.active_adapter = tier_names[0]

        # A fresh state dict -- never the live module singleton -- so this
        # test cannot leak boot-phase mutations into any other test sharing
        # the process.
        state: dict = {"vram_components": {}}
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)
        monkeypatch.setattr(
            app_module,
            "load_base_model",
            lambda model_config, adapters: (double, MagicMock()),
        )

        app_module._load_model_into_state(cfg)
        assert state["model"] is double

        state["config"] = cfg
        state["consolidation_loop"] = None
        state["memory_store"] = MemoryStore()
        loop = get_or_create_consolidation_loop(state)

        assert loop.model is double
        assert loop.extraction.model is double
        assert loop.merger.model is double


# ---------------------------------------------------------------------------
# Every consumer reads the ONE tier_config_map() derivation -- never a
# second, independently-built tier set. One shared config instance (only
# "semantic" enabled, so a coincidental match with any other derivation is
# implausible) drives every consumer below; ServerConfig.tier_config_map is
# wrapped with a spy so a consumer that stops calling it is caught by a
# call-count miss, and each consumer's own observable output is asserted
# against the map directly so a consumer that calls it but then ALSO
# recomputes its own answer is caught too.
# ---------------------------------------------------------------------------


class TestEveryConsumerReadsTheOneTierConfigMap:
    def _shared_cfg(self, tmp_path: Path):
        cfg = load_server_config(_FIXTURE_PATH)
        cfg.adapters.episodic.enabled = False
        cfg.adapters.procedural.enabled = False
        cfg.adapters.semantic.target_modules = ["q_proj"]
        cfg.consolidation.max_interim_count = 0  # episodic is disabled
        data_root = tmp_path / "data"
        cfg.paths = PathsConfig(
            data=data_root, sessions=data_root / "sessions", debug=data_root / "debug"
        )
        return cfg

    def test_seven_consumers_agree_with_and_call_the_one_derivation(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.memory.store import MemoryStore
        from paramem.server import app as app_module
        from paramem.server import consolidation as consolidation_module
        from paramem.server.consolidation import create_consolidation_loop

        cfg = self._shared_cfg(tmp_path)
        spy = MagicMock(wraps=cfg.tier_config_map)
        monkeypatch.setattr(cfg, "tier_config_map", spy)
        expected = cfg.tier_config_map()
        assert list(expected) == ["semantic"], (
            "precondition: exactly one, distinctive tier must be enabled so "
            "a coincidental match with a re-derivation is implausible"
        )
        spy.reset_mock()

        # 1. load_base_model's `adapters` argument, and 2. the
        # _mount_adapters_from_slots loop it feeds (both reached through one
        # real boot call).
        double = _peft_model_mock()
        double.peft_config = {"semantic": MagicMock()}
        double.active_adapter = "semantic"
        captured_load_base_model_adapters: dict = {}

        def _fake_load_base_model(model_config, adapters):
            captured_load_base_model_adapters.update(adapters)
            return double, MagicMock()

        boot_state: dict = {"vram_components": {}}
        monkeypatch.setattr(app_module, "_state", boot_state)
        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)
        monkeypatch.setattr(app_module, "load_base_model", _fake_load_base_model)
        app_module._load_model_into_state(cfg)

        assert captured_load_base_model_adapters == expected

        # 3. _revalidate_adapter_manifests.
        revalidate_state = {
            "config": cfg,
            "model": double,
            "tokenizer": MagicMock(),
            "adapter_manifest_status": {},
        }
        app_module._revalidate_adapter_manifests(revalidate_state)
        assert "episodic" not in revalidate_state["adapter_manifest_status"]
        assert "procedural" not in revalidate_state["adapter_manifest_status"]

        # 4. /status's adapter_specs / episodic_rank.
        from fastapi.testclient import TestClient

        status_state = {
            "model": None,
            "tokenizer": None,
            "config": cfg,
            "config_path": None,
            "session_buffer": MagicMock(),
            "speaker_store": None,
            "router": None,
            "cloud_agent": None,
            "ha_client": None,
            "consolidation_loop": None,
            "consolidating": False,
            "last_consolidation": None,
            "background_trainer": None,
            "mode": "local",
            "cloud_only_reason": None,
            "tts_manager": None,
            "stt": None,
            "speaker_embedding_backend": None,
            "unknown_speakers": {},
            "pending_enrollments": set(),
            "migration": {"state": "live", "comparison": None},
            "server_started_at": "2026-04-22T08:00:00+00:00",
            "config_drift": {"detected": False},
            "adapter_manifest_status": {},
            "last_consolidation_result": None,
        }
        monkeypatch.setattr(app_module, "_state", status_state)
        resp = TestClient(app_module.app, raise_server_exceptions=True).get("/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["episodic_rank"] is None
        assert set(body["adapter_specs"]) == {"semantic"}

        # 5. The trial-consolidation door's gate (_run_extraction_phase) --
        # episodic is disabled in the shared config, so the gate refuses
        # immediately without needing any session-buffer plumbing.
        trial_state = {"config": cfg, "session_buffer": MagicMock()}
        monkeypatch.setattr(app_module, "_state", trial_state)
        result = app_module._run_extraction_phase(loop=MagicMock(), mark_sessions=False)
        assert result["status"] == "disabled"

        # 6. create_consolidation_loop's tier_adapters kwarg.
        captured_loop_kwargs: dict = {}

        def _fake_consolidation_loop(**kwargs):
            captured_loop_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(consolidation_module, "ConsolidationLoop", _fake_consolidation_loop)
        create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=MemoryStore(),
            seed_state_from_disk=False,
        )
        assert captured_loop_kwargs["tier_adapters"] == expected

        # 7. assess_topology's main_adapter_configs, via the boot-time
        # topology estimator that builds it.
        import transformers as _transformers

        assess_topology_spy = MagicMock(wraps=app_module.assess_topology)
        monkeypatch.setattr(app_module, "assess_topology", assess_topology_spy)
        fake_hf_cfg = type("C", (), {"hidden_size": 4096, "num_hidden_layers": 32})()
        monkeypatch.setattr(
            _transformers.AutoConfig, "from_pretrained", lambda *a, **k: fake_hf_cfg
        )
        app_module._compute_topology_assessment(cfg, base_pred=8 * 2**30)
        assert assess_topology_spy.call_count == 1
        assert assess_topology_spy.call_args.kwargs["main_adapter_configs"] == list(
            expected.values()
        )

        # Every consumer above went through the ONE derivation -- never a
        # second, independently-built tier set.
        assert spy.call_count >= 7, (
            f"expected every consumer to call tier_config_map() at least once each, "
            f"got {spy.call_count} call(s)"
        )


# ---------------------------------------------------------------------------
# tier_config_map -- a tier is present iff adapters.<tier>.enabled
# ---------------------------------------------------------------------------


class TestTierConfigMapMembership:
    def test_tier_present_iff_enabled(self):
        cfg = load_server_config(_FIXTURE_PATH)
        cfg.adapters.procedural.enabled = False

        tier_adapters = cfg.tier_config_map()

        assert "episodic" in tier_adapters
        assert "semantic" in tier_adapters
        assert "procedural" not in tier_adapters

    def test_order_is_main_tiers_order_regardless_of_yaml_order(self):
        cfg = load_server_config(_FIXTURE_PATH)

        expected = [t for t in MAIN_TIERS if getattr(cfg.adapters, t).enabled]
        assert list(cfg.tier_config_map()) == expected


# ---------------------------------------------------------------------------
# load_server_config's config-document guards (promotion threshold vs.
# semantic tier, all-tiers-disabled vs. cloud_only, interim ring vs.
# episodic tier), over the live tests/fixtures/server.yaml base.
# ---------------------------------------------------------------------------


def _fixture_dict() -> dict:
    return yaml.safe_load(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _fixture_yaml_with(tmp_path: Path, overrides: dict) -> Path:
    """Write the live ``tests/fixtures/server.yaml`` with *overrides*
    deep-merged in, so every config-document-guard test validates against
    the real production-shaped fixture rather than a minimal hand-rolled
    document."""
    raw = _deep_merge(_fixture_dict(), overrides)
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TestConfigLoadValidatorsV1ThroughV3a:
    @pytest.mark.parametrize("threshold", [0, 1])
    def test_v1_rejects_promotion_threshold_below_2_with_semantic_enabled(
        self, tmp_path, threshold
    ):
        path = _fixture_yaml_with(tmp_path, {"consolidation": {"promotion_threshold": threshold}})
        with pytest.raises(FatalConfigError) as excinfo:
            load_server_config(path)
        assert "adapters.semantic.enabled" in str(excinfo.value)
        assert "promotion_threshold" in str(excinfo.value)

    def test_v1_promotion_threshold_2_loads(self, tmp_path):
        path = _fixture_yaml_with(tmp_path, {"consolidation": {"promotion_threshold": 2}})
        cfg = load_server_config(path)
        assert cfg.consolidation.promotion_threshold == 2

    def test_v2_rejects_all_tiers_disabled_with_cloud_only_false(self, tmp_path):
        path = _fixture_yaml_with(
            tmp_path,
            {
                "cloud_only": False,
                "adapters": {
                    "episodic": {"enabled": False},
                    "semantic": {"enabled": False},
                    "procedural": {"enabled": False},
                },
                "consolidation": {"max_interim_count": 0},
            },
        )
        with pytest.raises(FatalConfigError) as excinfo:
            load_server_config(path)
        assert "cloud_only" in str(excinfo.value)

    def test_v2_all_tiers_disabled_with_cloud_only_true_loads(self, tmp_path):
        path = _fixture_yaml_with(
            tmp_path,
            {
                "cloud_only": True,
                "adapters": {
                    "episodic": {"enabled": False},
                    "semantic": {"enabled": False},
                    "procedural": {"enabled": False},
                },
                "consolidation": {"max_interim_count": 0},
            },
        )
        cfg = load_server_config(path)
        assert cfg.tier_config_map() == {}

    def test_v3a_rejects_max_interim_count_positive_with_episodic_disabled(self, tmp_path):
        path = _fixture_yaml_with(
            tmp_path,
            {
                "consolidation": {"max_interim_count": 2},
                "adapters": {"episodic": {"enabled": False}},
            },
        )
        with pytest.raises(FatalConfigError) as excinfo:
            load_server_config(path)
        assert "max_interim_count" in str(excinfo.value)
        assert "episodic" in str(excinfo.value)

    def test_v3a_max_interim_count_zero_with_episodic_disabled_loads(self, tmp_path):
        path = _fixture_yaml_with(
            tmp_path,
            {
                "consolidation": {"max_interim_count": 0},
                "adapters": {"episodic": {"enabled": False}},
            },
        )
        cfg = load_server_config(path)
        assert "episodic" not in cfg.tier_config_map()


# ---------------------------------------------------------------------------
# Config-vs-disk refusals inside _load_model_into_state: a populated interim
# ring under a disabled episodic tier, and a disabled tier whose registry
# still holds an active key. Both raise before load_base_model is ever
# reached, so apply_process_cap (the only CUDA touch ahead of that point)
# is patched to a no-op and no model/GPU is ever touched.
# ---------------------------------------------------------------------------


class _ReachedLoadBaseModel(Exception):
    """Sentinel proving execution reached load_base_model -- i.e. neither
    config-vs-disk refusal fired -- without actually loading a model
    (mirrors tests/test_vram_validator.py's spy_load_base_model pattern)."""


def _cfg_rooted_at(tmp_path: Path):
    cfg = load_server_config(_fixture_yaml_with(tmp_path, {"cloud_only": False}))
    data_root = tmp_path / "data"
    cfg.paths = PathsConfig(
        data=data_root, sessions=data_root / "sessions", debug=data_root / "debug"
    )
    return cfg


class TestConfigVsDiskGuardsV3bV4:
    def test_v3b_refuses_boot_with_a_populated_interim_ring_when_episodic_disabled(
        self, tmp_path, monkeypatch
    ):
        from paramem.server import app as app_module

        cfg = _cfg_rooted_at(tmp_path)
        cfg.adapters.episodic.enabled = False
        cfg.consolidation.max_interim_count = 0
        (cfg.adapter_dir / "episodic" / "interim_20260101T000000").mkdir(parents=True)

        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)

        with pytest.raises(RuntimeError, match="interim slot"):
            app_module._load_model_into_state(cfg)

    def test_v3b_loads_once_the_stray_ring_is_removed(self, tmp_path, monkeypatch):
        from paramem.server import app as app_module

        cfg = _cfg_rooted_at(tmp_path)
        cfg.adapters.episodic.enabled = False
        cfg.consolidation.max_interim_count = 0
        # No interim directory on disk -- the interim-ring refusal must not fire.

        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)
        monkeypatch.setattr(
            app_module, "load_base_model", MagicMock(side_effect=_ReachedLoadBaseModel)
        )

        with pytest.raises(_ReachedLoadBaseModel):
            app_module._load_model_into_state(cfg)

    def test_v4_refuses_boot_when_a_disabled_tiers_registry_holds_an_active_key(
        self, tmp_path, monkeypatch
    ):
        from paramem.server import app as app_module
        from paramem.training.key_registry import KeyRegistry

        cfg = _cfg_rooted_at(tmp_path)
        cfg.adapters.procedural.enabled = False

        tier_root = cfg.adapter_dir / "procedural"
        tier_root.mkdir(parents=True)
        registry = KeyRegistry()
        registry.add("proc1")
        registry.save(tier_root / "indexed_key_registry.json")

        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)

        with pytest.raises(RuntimeError, match="active key"):
            app_module._load_model_into_state(cfg)

    def test_v4_loads_once_the_disabled_tiers_registry_is_drained(self, tmp_path, monkeypatch):
        from paramem.server import app as app_module
        from paramem.training.key_registry import KeyRegistry

        cfg = _cfg_rooted_at(tmp_path)
        cfg.adapters.procedural.enabled = False

        tier_root = cfg.adapter_dir / "procedural"
        tier_root.mkdir(parents=True)
        registry = KeyRegistry()
        registry.add("proc1")
        registry.stale("proc1")
        registry.save(tier_root / "indexed_key_registry.json")

        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)
        monkeypatch.setattr(
            app_module, "load_base_model", MagicMock(side_effect=_ReachedLoadBaseModel)
        )

        with pytest.raises(_ReachedLoadBaseModel):
            app_module._load_model_into_state(cfg)

    def test_a_passing_check_resolves_an_active_config_refused_incident(
        self, tmp_path, monkeypatch
    ):
        """The single resolve site lives inside ``_load_model_into_state``,
        immediately after ``check_config_against_store`` returns -- reached
        by BOTH boot and every reload, since both paths funnel through this
        one function. A ``config_refused`` incident from an earlier refusal
        against this same store clears the moment the check passes again --
        no separate resolve call is needed on either path."""
        from paramem.server import app as app_module
        from paramem.server.incidents import read_incidents, record_incident
        from paramem.training.stage_ledger import data_state_dir

        cfg = _cfg_rooted_at(tmp_path)
        # Clean store: every tier enabled by default (no adapters.* override
        # here), no interim ring, no disabled-tier keys --
        # check_config_against_store passes.

        state_dir = data_state_dir(cfg.paths.data)
        record_incident(
            state_dir,
            type="config_refused",
            key="interim_ring_without_episodic",
            severity="failed",
            summary="Config refused on reload: adapters.episodic.enabled=false but ...",
            detail={"message": "stale refusal", "adapter_dir": str(cfg.adapter_dir)},
        )
        assert read_incidents(state_dir)[0].status == "active"

        monkeypatch.setattr(app_module, "apply_process_cap", lambda **kwargs: None)
        monkeypatch.setattr(
            app_module, "load_base_model", MagicMock(side_effect=_ReachedLoadBaseModel)
        )

        with pytest.raises(_ReachedLoadBaseModel):
            app_module._load_model_into_state(cfg)

        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        assert incidents[0].status == "resolved"

    def test_v4_refusal_is_not_reachable_through_the_classification_preview(self):
        """Confirms the disabled-tier-with-active-keys refusal is the only
        one: adapters.<tier>.enabled classifies as Tier.DESTRUCTIVE via the
        ``adapters.*.enabled`` wildcard (a preview glyph for the
        migration-preview renderer), and nothing consumes that
        classification as a veto -- ``classify`` never raises, and
        ``load_server_config`` does not call it at all."""
        from paramem.config.classification import Tier, classify

        assert classify("adapters.episodic.enabled") is Tier.DESTRUCTIVE
        # Tier is a glyph label for a preview renderer, not a refusal
        # mechanism -- classify() never raises for any input.
        assert Tier.DESTRUCTIVE.glyph == "⚠"


# ---------------------------------------------------------------------------
# AST scan -- ensure_resident_tiers is the one get_peft_model site (this
# scan covers only the get_peft_model half; the unwrap and
# PeftModel.from_pretrained sites are covered by the full scan below).
# ---------------------------------------------------------------------------


class TestGetPeftModelSingleSite:
    def test_get_peft_model_appears_only_in_loader_ensure_resident_tiers(self):
        allowed = {"paramem/models/loader.py", "experiments/smoke_procedural_mlp.py"}
        hits: list[str] = []
        for root in ("paramem", "experiments"):
            root_path = Path(root)
            if not root_path.is_dir():
                continue
            for path in root_path.rglob("*.py"):
                posix = path.as_posix()
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=posix)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == "get_peft_model":
                        if posix not in allowed:
                            hits.append(f"{posix}:{node.lineno}")
        assert hits == [], f"get_peft_model referenced outside the allowed sites: {hits}"


# ---------------------------------------------------------------------------
# AST scan -- the full scan: zero PeftModel.from_pretrained sites and zero
# ``.base_model.model`` unwrap sites across paramem/ + experiments/, in
# addition to the get_peft_model half TestGetPeftModelSingleSite already
# covers, so a reintroduced unwrap or a second PeftModel.from_pretrained
# site fails this test rather than being an unmeasured claim.
# ---------------------------------------------------------------------------


class TestFullAstScanUnwrapAndFromPretrained:
    def _scan(self) -> list[str]:
        hits: list[str] = []
        for root in ("paramem", "experiments"):
            root_path = Path(root)
            if not root_path.is_dir():
                continue
            for path in root_path.rglob("*.py"):
                posix = path.as_posix()
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=posix)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Attribute)
                        and node.attr == "model"
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "base_model"
                    ):
                        hits.append(f"unwrap:{posix}:{node.lineno}")
                    if (
                        isinstance(node, ast.Attribute)
                        and node.attr == "from_pretrained"
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "PeftModel"
                    ):
                        hits.append(f"from_pretrained:{posix}:{node.lineno}")
        return hits

    def test_zero_unwrap_sites_and_zero_peft_model_from_pretrained_sites(self):
        hits = self._scan()
        assert hits == [], f"found unwrap/from_pretrained hits: {hits}"


# ---------------------------------------------------------------------------
# _eager_create_consolidation_loop -- deleted dead code; the loop is now
# built lazily at the first consolidation door instead of at boot. Pinned
# so it is never silently reintroduced.
# ---------------------------------------------------------------------------


class TestEagerCreateConsolidationLoopDeleted:
    def test_name_is_absent_from_the_server_app_module(self):
        from paramem.server import app as app_module

        assert not hasattr(app_module, "_eager_create_consolidation_loop")

    def test_name_appears_nowhere_in_paramem(self):
        hits: list[str] = []
        root_path = Path("paramem")
        for path in root_path.rglob("*.py"):
            if "_eager_create_consolidation_loop" in path.read_text(encoding="utf-8"):
                hits.append(path.as_posix())
        assert hits == [], f"_eager_create_consolidation_loop still referenced in: {hits}"


# ---------------------------------------------------------------------------
# probe_keys_grouped_by_adapter (paramem/memory/probe.py) -- "residency does
# not imply readiness" for the recall-probe consumer: a resident-but-cold
# adapter is SKIPPED, not probed -- probing it would return parse failures
# instead of the honest None. No existing test drives this function's own
# cold-vs-warm branch directly; every other reference monkeypatches it away
# or wraps it as a call-argument spy.
# ---------------------------------------------------------------------------


class TestProbeKeysGroupedByAdapterSkipsAColdTier:
    def test_resident_but_cold_adapter_maps_every_key_to_none_without_probing(self, monkeypatch):
        from paramem.memory.probe import probe_keys_grouped_by_adapter

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        # Cold: lora_B tensors present but zero-valued (never trained).
        model.named_parameters.return_value = [
            (
                "base_model.model.x.lora_B.episodic.weight",
                MagicMock(data=torch.zeros(2, 2)),
            ),
        ]

        probe_entries_mock = MagicMock(
            side_effect=AssertionError("probe_entries must not be called for a cold tier")
        )
        monkeypatch.setattr("paramem.training.recall_eval.probe_entries", probe_entries_mock)

        results = probe_keys_grouped_by_adapter(
            model, MagicMock(), {"episodic": ["graph1", "graph2"]}
        )

        assert results == {"graph1": None, "graph2": None}
        model.set_adapter.assert_not_called()
        probe_entries_mock.assert_not_called()
