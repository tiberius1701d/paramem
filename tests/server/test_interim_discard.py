"""Tests for ``POST /interim/discard``.

Mocked — no GPU, no real model. Mirrors the mocking style of
``tests/server/test_speaker_forget.py`` and ``tests/server/test_gates.py``.

Coverage
--------
- Preview (``confirm`` omitted/false): 409 (owner-ruled
  ``_INTERIM_DISCARD_UNCONFIRMED_STATUS``), inventory in ``would_discard``,
  no mutation.
- Happy path: real store tiers + on-disk dirs + PEFT names all reaped;
  main tiers untouched; response counts match the pre-mutation inventory.
- Empty ring: 200 ``noop_empty_ring``, nothing touched.
- Guard matrix: consolidating / bg training / cloud-only / trial-active all
  409 with the mapped error, no mutation.
- Operation order: drop_tier -> unload_interim_adapters -> save_key_metadata
  -> router.reload.
- Ring incidents resolved with a non-null ``resolved_reason``; unrelated
  incident types untouched.
- ``adapter_manifest_status`` rows for discarded names popped; surviving
  rows (main tiers) untouched.
- ``record_last_run`` called with ``op_type="consolidation"``,
  ``outcome="interim_discarded"``.
- Non-PEFT (simulate) venue: bare object model, PEFT half skipped.
- Failure path: 500 propagates, ``_state["consolidating"]`` cleared in the
  ``finally``, ``_state["last_consolidation"]`` left unstamped.
- ``_state["consolidating"]`` is True for the duration of the mutation.
- Auth: ``/interim/discard`` carries ``require_admin`` in the real route
  table (also covered by ``tests/server/test_require_admin.py``).
- ``_state["last_consolidation"]`` stamped on a confirmed discard, left
  ``None`` on a no-op; a ``record_last_run`` failure is swallowed
  (logged, not raised) and the stamp/router-reload tail still runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from peft import PeftModel

import paramem.server.app as app_module
from paramem.memory.store import MemoryStore
from paramem.server.incidents import read_incidents, record_incident
from paramem.server.run_status import read_last_runs

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _entry(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> dict:
    return {"key": key, "subject": subject, "predicate": predicate, "object": obj}


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.key_metadata_path = tmp_path / "registry" / "key_metadata.json"
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _make_peft_model(*adapter_names: str) -> MagicMock:
    """Stub PeftModel — mirrors ``_make_stub_peft_model`` in
    ``tests/test_interim_adapter_lifecycle.py``: a real dict-backed
    ``peft_config`` so membership/iteration/deletion behave like PEFT."""
    model = MagicMock(spec=PeftModel)
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _make_loop(store: MemoryStore, model) -> MagicMock:
    loop = MagicMock()
    loop.store = store
    loop.model = model
    loop.promoted_keys = set()
    loop.cycle_count = 0
    loop.trial_key_metadata_path = None
    return loop


def _seed_interim_slot(store: MemoryStore, adapter_dir: Path, stamp: str, key: str) -> str:
    """Seed one interim tier in the store, its on-disk dir, and bookkeeping.

    Returns the interim tier/adapter name.
    """
    tier = f"episodic_interim_{stamp}"
    store.put(tier, key, _entry(key), simhash=1)
    store.set_bookkeeping(key, speaker_id="speaker0", relation_type="episodic", first_seen="t0")
    (adapter_dir / "episodic" / f"interim_{stamp}").mkdir(parents=True, exist_ok=True)
    return tier


def _seed_main_tier(store: MemoryStore, adapter_dir: Path, tier: str, key: str) -> None:
    store.put(tier, key, _entry(key), simhash=2)
    store.set_bookkeeping(key, speaker_id="speaker0", relation_type="episodic", first_seen="t0")
    tier_dir = adapter_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    store.registry(tier).save(tier_dir / "indexed_key_registry.json")


def _make_state(
    tmp_path: Path,
    *,
    loop=None,
    config=None,
    mode: str = "local",
    consolidating: bool = False,
    background_trainer=None,
    migration=None,
    router=None,
    session_buffer=None,
) -> dict:
    return {
        "config": config or _make_config(tmp_path),
        "consolidation_loop": loop,
        "mode": mode,
        "consolidating": consolidating,
        "background_trainer": background_trainer,
        "migration": migration,
        "router": router or MagicMock(),
        "adapter_manifest_status": {},
        "session_buffer": session_buffer or MagicMock(),
        "last_consolidation": None,
    }


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Preview: confirm omitted/false
# ---------------------------------------------------------------------------


class TestPreview:
    def test_unconfirmed_returns_owner_ruled_status_with_inventory(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={})

        assert resp.status_code == app_module._INTERIM_DISCARD_UNCONFIRMED_STATUS, resp.text
        detail = resp.json()["detail"]
        assert detail["error"] == "confirmation_required"
        would_discard = detail["would_discard"]
        assert would_discard["store_tiers"] == ["episodic_interim_20260801T0000"]
        assert would_discard["active_keys"] == {"episodic_interim_20260801T0000": 1}

        # Nothing mutated.
        assert "episodic_interim_20260801T0000" in store.tiers_with_registry()
        assert (cfg.adapter_dir / "episodic" / "interim_20260801T0000").exists()
        assert state["consolidating"] is False

    def test_explicit_confirm_false_is_still_unconfirmed(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": False})

        assert resp.status_code == app_module._INTERIM_DISCARD_UNCONFIRMED_STATUS


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_confirmed_discard_reaps_the_whole_ring(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        _seed_interim_slot(store, cfg.adapter_dir, "20260802T0000", "graph2")
        model = _make_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260801T0000",
            "episodic_interim_20260802T0000",
        )
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "discarded"
        assert sorted(body["discarded_tiers"]) == [
            "episodic_interim_20260801T0000",
            "episodic_interim_20260802T0000",
        ]
        assert sorted(body["unloaded_adapters"]) == [
            "episodic_interim_20260801T0000",
            "episodic_interim_20260802T0000",
        ]
        assert sorted(body["removed_dirs"]) == ["interim_20260801T0000", "interim_20260802T0000"]
        assert body["active_keys_destroyed"] == {
            "episodic_interim_20260801T0000": 1,
            "episodic_interim_20260802T0000": 1,
        }

        # Store: both interim tiers gone.
        assert "episodic_interim_20260801T0000" not in store.tiers_with_registry()
        assert "episodic_interim_20260802T0000" not in store.tiers_with_registry()
        assert store.bookkeeping_for_key("graph1") is None
        assert store.bookkeeping_for_key("graph2") is None

        # PEFT: interim names gone, mains survive.
        assert "episodic_interim_20260801T0000" not in model.peft_config
        assert "episodic_interim_20260802T0000" not in model.peft_config
        assert "episodic" in model.peft_config

        # Disk: interim dirs gone.
        assert not (cfg.adapter_dir / "episodic" / "interim_20260801T0000").exists()
        assert not (cfg.adapter_dir / "episodic" / "interim_20260802T0000").exists()

        # Router reloaded; consolidating cleared.
        state["router"].reload.assert_called_once_with()
        assert state["consolidating"] is False

        # last_consolidation stamped on the event loop (GET /status must not
        # show a fresh last_consolidation_result beside a stale timestamp).
        # Parse (not just isinstance-check) so a malformed non-ISO string
        # would fail this test rather than pass it.
        assert state["last_consolidation"] is not None
        datetime.fromisoformat(state["last_consolidation"])

    def test_main_tiers_untouched(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        _seed_main_tier(store, cfg.adapter_dir, "episodic", "graph-main")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert "episodic" in store.tiers_with_registry()
        assert store.get("graph-main") == _entry("graph-main")
        assert (cfg.adapter_dir / "episodic" / "indexed_key_registry.json").exists()
        assert "episodic" in model.peft_config

    def test_pending_sessions_untouched(self, tmp_path, monkeypatch):
        """Pending sessions in the SessionBuffer are never in an interim slot
        (absorbed by the next fold) — the discard door must not touch them."""
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        session_buffer = MagicMock()
        state = _make_state(tmp_path, loop=loop, config=cfg, session_buffer=session_buffer)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        session_buffer.discard_sessions.assert_not_called()
        session_buffer.mark_consolidated.assert_not_called()


# ---------------------------------------------------------------------------
# Stray on-disk directories
# ---------------------------------------------------------------------------


class TestStrayInvalidStampDir:
    def test_stray_invalid_stamp_dir_still_discarded(self, tmp_path, monkeypatch):
        """A directory like ``interim_garbage/`` (invalid stamp suffix) has
        no store registration but is still a real on-disk interim directory
        — the endpoint must still discard it (the reaper would happily
        remove it) rather than 500 on a ``ValueError`` from re-deriving its
        path via ``interim_dir_for_name``."""
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        model = _make_peft_model("episodic")
        loop = _make_loop(store, model)
        stray = cfg.adapter_dir / "episodic" / "interim_garbage"
        stray.mkdir(parents=True)
        (stray / "graph.json").write_text("{}")
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "discarded"
        assert body["removed_dirs"] == ["interim_garbage"]
        assert not stray.exists()


# ---------------------------------------------------------------------------
# promoted_keys pruning
# ---------------------------------------------------------------------------


class TestPromotedKeysPruned:
    def test_discarded_tier_keys_pruned_from_promoted_keys(self, tmp_path, monkeypatch):
        """The discard door must prune ``loop.promoted_keys`` of the
        discarded tiers' keys before rewriting ``key_metadata.json`` — the
        same defect the forget handler fixes via
        ``promoted_keys.difference_update``.  A key from a surviving main
        tier must not be pruned."""
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        _seed_main_tier(store, cfg.adapter_dir, "episodic", "graph-main")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        loop.promoted_keys = {"graph1", "graph-main"}
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert loop.promoted_keys == {"graph-main"}, (
            "discarded tier's key must be pruned from promoted_keys; "
            "surviving main-tier key must remain"
        )


# ---------------------------------------------------------------------------
# Empty ring
# ---------------------------------------------------------------------------


class TestEmptyRing:
    def test_noop_when_nothing_to_discard(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        model = _make_peft_model("episodic", "semantic", "procedural")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "noop_empty_ring"
        assert body["discarded_tiers"] == []
        assert body["unloaded_adapters"] == []
        assert body["removed_dirs"] == []
        assert body["resolved_incidents"] == 0

        # key_metadata.json was never written.
        assert not cfg.key_metadata_path.exists()
        state["router"].reload.assert_not_called()
        assert state["last_consolidation"] is None

    def test_noop_also_short_circuits_unconfirmed_request(self, tmp_path, monkeypatch):
        """An empty ring returns the noop 200 even without confirm=true — a
        no-op is the answer, not a refusal."""
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        model = _make_peft_model("episodic")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={})

        assert resp.status_code == 200
        assert resp.json()["status"] == "noop_empty_ring"


# ---------------------------------------------------------------------------
# Guard matrix
# ---------------------------------------------------------------------------


class TestGuardMatrix:
    def _assert_refused_without_mutation(self, monkeypatch, tmp_path, state):
        loop = MagicMock()
        state["consolidation_loop"] = loop
        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})
        return resp, loop

    def test_consolidating_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, consolidating=True)
        resp, loop = self._assert_refused_without_mutation(monkeypatch, tmp_path, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidating"
        loop.store.drop_tier.assert_not_called()

    def test_bg_training_refuses_409(self, tmp_path, monkeypatch):
        trainer = MagicMock()
        trainer.is_training = True
        state = _make_state(tmp_path, background_trainer=trainer)
        resp, loop = self._assert_refused_without_mutation(monkeypatch, tmp_path, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "training_active"
        loop.store.drop_tier.assert_not_called()

    def test_cloud_only_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="cloud-only")
        resp, loop = self._assert_refused_without_mutation(monkeypatch, tmp_path, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "cloud_only"
        loop.store.drop_tier.assert_not_called()

    def test_trial_active_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, migration={"state": "TRIAL"})
        resp, loop = self._assert_refused_without_mutation(monkeypatch, tmp_path, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"
        loop.store.drop_tier.assert_not_called()

    def test_base_swap_active_refuses_409(self, tmp_path, monkeypatch):
        """migration.base_swap_active=True refuses 409, same as the other
        four guard verdicts — closes the window where a base-swap
        orchestration's direct (non-mode-flipping) dispatch could win
        ``gpu_lock`` ahead of this door and release + recreate the
        ConsolidationLoop this door already captured a reference to.
        """
        state = _make_state(tmp_path, migration={"base_swap_active": True})
        resp, loop = self._assert_refused_without_mutation(monkeypatch, tmp_path, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "base_swap_active"
        loop.store.drop_tier.assert_not_called()


# ---------------------------------------------------------------------------
# Loop staleness across gpu_lock — a config-apply (or, before this fix, a
# base-swap orchestration racing ahead of the base_swap_active guard) that
# releases + recreates the ConsolidationLoop while this door holds gpu_lock
# must not leave the sync worker mutating the released pre-lock loop.
# ---------------------------------------------------------------------------


class TestLoopStalenessAcrossLock:
    def test_discard_sync_mutates_the_fresh_loop_not_the_pre_lock_capture(
        self, tmp_path, monkeypatch
    ):
        """Swap ``_state['consolidation_loop']`` to a fresh mock while the
        handler holds ``gpu_lock`` (simulating a config-apply that raced
        ahead of this door and won the lock first) — the executor-side
        mutation (``store.drop_tier`` / PEFT unmount) must land on the FRESH
        loop's store/model, not the loop the handler captured before the
        lock, which proves ``_discard_sync`` re-resolves ``loop`` as its
        first statement rather than closing over the pre-lock capture.
        """
        import contextlib

        import paramem.server.gpu_lock as gpu_lock_module

        cfg = _make_config(tmp_path)
        tier = "episodic_interim_20260801T0000"

        store_a = MemoryStore(replay_enabled=True)  # the handler's pre-lock capture
        _seed_interim_slot(store_a, cfg.adapter_dir, "20260801T0000", "graph1")
        model_a = _make_peft_model("episodic", tier)
        loop_a = _make_loop(store_a, model_a)

        store_b = MemoryStore(replay_enabled=True)  # swapped in while the lock is held
        _seed_interim_slot(store_b, cfg.adapter_dir, "20260801T0000", "graph1")
        model_b = _make_peft_model("episodic", tier)
        loop_b = _make_loop(store_b, model_b)

        state = _make_state(tmp_path, loop=loop_a, config=cfg)

        @contextlib.asynccontextmanager
        async def _swap_loop_gpu_lock():
            state["consolidation_loop"] = loop_b
            yield

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(gpu_lock_module, "gpu_lock", _swap_loop_gpu_lock)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        # The pre-lock capture is untouched...
        assert tier in store_a.tiers_with_registry()
        assert tier in model_a.peft_config
        # ...the fresh (swapped-in) loop is the one actually mutated.
        assert tier not in store_b.tiers_with_registry()
        assert tier not in model_b.peft_config

    def test_discard_sync_mutates_the_fresh_config_not_the_pre_lock_capture(
        self, tmp_path, monkeypatch
    ):
        """Same race, but on ``_state['config']`` rather than the loop: a
        config-apply that wins ``gpu_lock`` first also replaces
        ``_state["config"]``.  ``_discard_sync``'s on-disk deletion
        (``unload_interim_adapters(loop.model, config.adapter_dir)``) must
        target the FRESH config's ``adapter_dir``, not the pre-lock one —
        proves the sync worker re-resolves ``config`` too, not just ``loop``.
        """
        import contextlib

        import paramem.server.gpu_lock as gpu_lock_module

        tier = "episodic_interim_20260801T0000"

        cfg_a = _make_config(tmp_path / "a")  # the handler's pre-lock capture
        store_a = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store_a, cfg_a.adapter_dir, "20260801T0000", "graph1")
        model_a = _make_peft_model("episodic", tier)
        loop_a = _make_loop(store_a, model_a)

        cfg_b = _make_config(tmp_path / "b")  # swapped in while the lock is held
        store_b = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store_b, cfg_b.adapter_dir, "20260801T0000", "graph1")
        model_b = _make_peft_model("episodic", tier)
        loop_b = _make_loop(store_b, model_b)

        state = _make_state(tmp_path, loop=loop_a, config=cfg_a)

        @contextlib.asynccontextmanager
        async def _swap_config_and_loop_gpu_lock():
            state["config"] = cfg_b
            state["consolidation_loop"] = loop_b
            yield

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(gpu_lock_module, "gpu_lock", _swap_config_and_loop_gpu_lock)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        # The pre-lock config's on-disk slot is untouched...
        assert (cfg_a.adapter_dir / "episodic" / "interim_20260801T0000").exists()
        assert tier in model_a.peft_config
        # ...the fresh (swapped-in) config's on-disk slot is the one reaped.
        assert not (cfg_b.adapter_dir / "episodic" / "interim_20260801T0000").exists()
        assert tier not in model_b.peft_config


# ---------------------------------------------------------------------------
# Operation order
# ---------------------------------------------------------------------------


class TestOperationOrder:
    def test_drop_tier_then_unload_then_save_metadata_then_router_reload(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        call_order: list[str] = []

        orig_drop_tier = store.drop_tier

        def _spy_drop_tier(tier):
            call_order.append("drop_tier")
            return orig_drop_tier(tier)

        monkeypatch.setattr(store, "drop_tier", _spy_drop_tier)

        orig_delete_adapter_side_effect = model.delete_adapter.side_effect
        tiers_at_unload: list[list[str]] = []

        def _spy_delete_adapter(name):
            if not call_order or call_order[-1] != "unload_interim_adapters":
                call_order.append("unload_interim_adapters")
                # Step 1 (drop_tier) must have already run by the time Step 2
                # (unload_interim_adapters) starts deleting PEFT adapters —
                # the tier becomes unroutable in RAM before its on-disk/PEFT
                # remnants are reaped.
                tiers_at_unload.append(list(store.tiers_with_registry()))
            return orig_delete_adapter_side_effect(name)

        model.delete_adapter.side_effect = _spy_delete_adapter

        def _spy_reload():
            call_order.append("router_reload")

        state["router"].reload.side_effect = _spy_reload

        import paramem.server.consolidation as consolidation_module

        orig_save_key_metadata = consolidation_module._save_key_metadata

        def _spy_save_key_metadata(loop_arg, config_arg):
            call_order.append("save_key_metadata")
            return orig_save_key_metadata(loop_arg, config_arg)

        monkeypatch.setattr(consolidation_module, "_save_key_metadata", _spy_save_key_metadata)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert tiers_at_unload, "unload_interim_adapters must have deleted at least one adapter"
        assert "episodic_interim_20260801T0000" not in tiers_at_unload[0], (
            "interim tier must already be dropped from the store's registry set "
            "by the time unload_interim_adapters runs"
        )
        assert call_order == [
            "drop_tier",
            "unload_interim_adapters",
            "save_key_metadata",
            "router_reload",
        ]


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------


class TestIncidentResolution:
    def test_ring_incidents_resolved_others_untouched(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        state_dir = cfg.paths.data / "state"
        for t in (
            "full_consolidation_overdue",
            "interim_cap_reached",
            "interim_overflow_pending",
            "vram_exhausted",
        ):
            record_incident(
                state_dir,
                type=t,
                key="k1",
                severity="warning",
                summary="test",
                detail={},
            )

        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert resp.json()["resolved_incidents"] == 3

        incidents = {i.type: i for i in read_incidents(state_dir)}
        for t in ("full_consolidation_overdue", "interim_cap_reached", "interim_overflow_pending"):
            assert incidents[t].status == "resolved"
            assert incidents[t].resolved_reason is not None
        assert incidents["vram_exhausted"].status == "active"


# ---------------------------------------------------------------------------
# adapter_manifest_status
# ---------------------------------------------------------------------------


class TestManifestStatusCleanup:
    def test_discarded_row_popped_surviving_row_kept(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)
        state["adapter_manifest_status"] = {
            "episodic_interim_20260801T0000": {"status": "unhealthy"},
            "episodic": {"status": "healthy"},
        }

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert "episodic_interim_20260801T0000" not in state["adapter_manifest_status"]
        assert "episodic" in state["adapter_manifest_status"]


# ---------------------------------------------------------------------------
# run_status
# ---------------------------------------------------------------------------


class TestRunStatusRecorded:
    def test_record_last_run_called_with_interim_discarded_outcome(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        last_runs = read_last_runs(cfg.paths.data / "state")
        rec = last_runs["consolidation"]
        assert rec.op_type == "consolidation"
        assert rec.outcome == "interim_discarded"


# ---------------------------------------------------------------------------
# Non-PEFT (simulate) venue
# ---------------------------------------------------------------------------


class TestNonPeftVenue:
    def test_bare_model_skips_peft_half_but_still_reaps_disk_and_store(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        loop = _make_loop(store, object())
        state = _make_state(tmp_path, loop=loop, config=cfg)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "discarded"
        assert body["unloaded_adapters"] == []
        assert "episodic_interim_20260801T0000" not in store.tiers_with_registry()
        assert not (cfg.adapter_dir / "episodic" / "interim_20260801T0000").exists()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailurePath:
    def test_reaper_failure_returns_500_and_clears_consolidating(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        import paramem.memory.interim_adapter as interim_adapter_module

        def _boom(*args, **kwargs):
            raise RuntimeError("reap failed")

        monkeypatch.setattr(interim_adapter_module, "unload_interim_adapters", _boom)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 500
        assert state["consolidating"] is False
        # The executor call raised before the coroutine ever reached the
        # no-await tail, so the stamp must not have run.
        assert state["last_consolidation"] is None

    def test_consolidating_is_true_during_the_operation(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        observed: list[bool] = []

        import paramem.memory.interim_adapter as interim_adapter_module

        orig_unload = interim_adapter_module.unload_interim_adapters

        def _spy_unload(model_arg, adapter_dir_arg):
            observed.append(state["consolidating"])
            return orig_unload(model_arg, adapter_dir_arg)

        monkeypatch.setattr(interim_adapter_module, "unload_interim_adapters", _spy_unload)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert observed == [True]
        assert state["consolidating"] is False

    def test_run_status_write_failure_does_not_500(self, tmp_path, monkeypatch):
        """A ``record_last_run`` failure must not turn an already-completed
        destructive ring drop into an HTTP 500 — matches the six finalizers
        (e.g. ``_finalize_interim``), which all wrap the call in
        ``try/except Exception: logger.exception(...)``.  Pre-fix, this call
        was the only one of seven writers left unwrapped, so this raise
        propagated straight through to a 500 despite the mutation having
        already fully succeeded.
        """
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        def _boom(*args, **kwargs):
            raise RuntimeError("run_status.json write failed")

        monkeypatch.setattr(app_module, "record_last_run", _boom)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "discarded"
        assert state["consolidating"] is False
        # The mutation still happened despite the bookkeeping failure.
        assert "episodic_interim_20260801T0000" not in store.tiers_with_registry()
        # The swallow happens inside _discard_sync (Step 6); the coroutine's
        # no-await tail (stamp + router.reload) still runs afterward — the
        # bookkeeping failure must not skip it.
        assert state["last_consolidation"] is not None
        datetime.fromisoformat(state["last_consolidation"])

    def test_save_key_metadata_failure_does_not_500(self, tmp_path, monkeypatch):
        """A ``_save_key_metadata`` failure (Step 3) must not turn an
        already-completed destructive ring drop into an HTTP 500 — Steps 1-2
        already dropped the tier from the store and reaped its adapter by
        the time this bookkeeping step runs.
        """
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        def _boom(*args, **kwargs):
            raise RuntimeError("key_metadata.json write failed")

        # _save_key_metadata is imported locally inside the handler on every
        # call (from paramem.server.consolidation) — patch the source, not
        # an app_module attribute.
        monkeypatch.setattr("paramem.server.consolidation._save_key_metadata", _boom)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "discarded"
        assert state["consolidating"] is False
        # The mutation still happened despite the bookkeeping failure.
        assert "episodic_interim_20260801T0000" not in store.tiers_with_registry()
        # The tail (stamp + router reload) still ran despite the Step 3 failure.
        assert state["last_consolidation"] is not None
        datetime.fromisoformat(state["last_consolidation"])
        state["router"].reload.assert_called_once()

    def test_resolve_incidents_failure_does_not_500(self, tmp_path, monkeypatch):
        """A ``resolve_incidents_by_type`` failure (Step 4) must not turn an
        already-completed destructive ring drop into an HTTP 500.
        """
        cfg = _make_config(tmp_path)
        store = MemoryStore(replay_enabled=True)
        _seed_interim_slot(store, cfg.adapter_dir, "20260801T0000", "graph1")
        model = _make_peft_model("episodic", "episodic_interim_20260801T0000")
        loop = _make_loop(store, model)
        state = _make_state(tmp_path, loop=loop, config=cfg)

        def _boom(*args, **kwargs):
            raise RuntimeError("incidents.json write failed")

        monkeypatch.setattr(app_module, "resolve_incidents_by_type", _boom)

        client = _make_client(monkeypatch, state)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "discarded"
        assert state["consolidating"] is False
        assert "episodic_interim_20260801T0000" not in store.tiers_with_registry()
        assert state["last_consolidation"] is not None
        datetime.fromisoformat(state["last_consolidation"])
        state["router"].reload.assert_called_once()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuthDependency:
    def test_interim_discard_has_require_admin(self):
        from paramem.server.app import app, require_admin

        route_map = {
            route.path: route
            for route in app.routes
            if hasattr(route, "path") and hasattr(route, "dependant")
        }
        route = route_map.get("/interim/discard")
        assert route is not None, "/interim/discard not found in app routes"
        dep_callables = [d.call for d in route.dependant.dependencies]
        assert require_admin in dep_callables, (
            "/interim/discard must carry require_admin dependency"
        )
