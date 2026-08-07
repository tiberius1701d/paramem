"""Tests for ``POST /debug/erase-keys``.

Mocked — no GPU, no real model, no server stop/start, no ``data/ha``.
Mirrors the mocking style of ``tests/server/test_speaker_forget.py`` (same
``_erase_keys_with_reap`` sequence, shared by both routes) and the gating
style of ``tests/server/test_debug_dump_endpoint.py``.

Coverage
--------
- Confirmation gate: ``confirm`` omitted/``False`` -> 409
  ``confirmation_required`` with a ``would_erase`` preview, no mutation —
  mirrors ``POST /interim/discard``'s Step 0d.
- Gating: ``config.debug=False`` -> 403 ``forbidden_not_debug``; auth
  (``require_admin`` in the real route table) is covered here AND via
  ``tests/server/test_require_admin.py``'s ``_ADMIN_PATHS`` introspection.
- Request validation: empty ``keys`` list and a blank-string entry -> 400
  ``invalid_keys``, no store mutation.
- The repair case: a key with a store/registry entry but NO bookkeeping
  record (unreachable via ``/speaker/forget``, which resolves its key set
  from ``store.iter_bookkeeping()``) is erased successfully.
- Mixed known/unknown request: known keys erased, unknown keys reported in
  ``unknown`` — never an error.
- Post-erase bookkeeping: ``key_metadata.json`` rewritten,
  ``loop.promoted_keys`` cleaned of erased keys.
- A tier the erase reduces to zero known keys is reaped (live PEFT unmount +
  on-disk artifact removal) via the same ``_reap_emptied_tiers`` path
  ``/speaker/forget`` uses.
- ``malformed_tier_name`` -> 500, no mutation (shared helper's pre-mutation
  abort, pinned via this door too).
- Busy guard: the same ``_consolidation_dispatch_guards`` verdicts
  ``/speaker/forget`` refuses under (consolidating / bg-training /
  cloud-only / trial-active / base-swap-active) refuse with 409 and no
  mutation.
- Router reloaded exactly once after a mutating request.
- Response store counts (``total``/``tiers``/``bookkeeping_total``) mirror
  ``GET /debug/dump``'s vocabulary and reflect the post-erase state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from peft import PeftModel

import paramem.server.app as app_module
from paramem.memory.entry import compute_simhash
from paramem.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers / fixtures — mirrors tests/server/test_speaker_forget.py
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, debug: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.key_metadata_path = tmp_path / "registry" / "key_metadata.json"
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _make_peft_model(*adapter_names: str) -> MagicMock:
    """Stub PeftModel — real dict-backed ``peft_config`` so membership /
    iteration / deletion behave like PEFT (mirrors ``_make_peft_model`` in
    ``tests/server/test_speaker_forget.py``)."""
    model = MagicMock(spec=PeftModel)
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _make_loop_with_store(store: MemoryStore) -> MagicMock:
    """Build a MagicMock ConsolidationLoop wrapping a real MemoryStore.

    Pre-wires ``promoted_keys`` as a real ``set`` — the handler calls
    ``loop.promoted_keys.difference_update(erased_keys)`` directly — plus a
    bare ``model``/``ensure_adapters`` so the reap path
    (``_reap_emptied_tiers``) has something to call without crashing on
    tests that never trigger a reap.  ``loop.write_key_metadata()`` is
    itself a mock on this ``MagicMock`` loop and never touches the
    filesystem.
    """
    loop = MagicMock()
    loop.store = store
    loop.promoted_keys = set()
    loop.model = MagicMock()
    loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)
    return loop


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
    adapter_manifest_status=None,
) -> dict:
    return {
        "config": config or _make_config(tmp_path),
        "consolidation_loop": loop,
        "mode": mode,
        "consolidating": consolidating,
        "background_trainer": background_trainer,
        "migration": migration,
        "router": router or MagicMock(),
        "adapter_manifest_status": (
            adapter_manifest_status if adapter_manifest_status is not None else {}
        ),
        "model": loop.model if loop is not None else MagicMock(),
        "tokenizer": None,
        "memory_store": loop.store if loop is not None else MagicMock(),
    }


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


def _seed_key(
    store: MemoryStore,
    tier: str,
    key: str,
    *,
    with_bookkeeping: bool,
    subject: str = "Alice",
    predicate: str = "lives_in",
    obj: str = "Berlin",
) -> None:
    """Seed a real known key (registry + entry + simhash) in *tier*.

    ``with_bookkeeping=False`` reproduces the repair case this endpoint
    exists for: registry-known, bookkeeping-absent.
    """
    fp = compute_simhash(key, subject, predicate, obj)
    store.put(
        tier,
        key,
        {"key": key, "subject": subject, "predicate": predicate, "object": obj},
        simhash=fp,
    )
    if with_bookkeeping:
        store.set_bookkeeping(key, speaker_id="speaker0", relation_type="episodic", first_seen="t0")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TestGating:
    def test_debug_false_returns_403(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        loop = _make_loop_with_store(store)
        cfg = _make_config(tmp_path, debug=False)
        state = _make_state(tmp_path, loop=loop, config=cfg)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"]})

        assert resp.status_code == 403
        assert resp.json()["status"] == "forbidden_not_debug"

    def test_empty_keys_returns_400(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        _seed_key(store, "episodic", "graph1", with_bookkeeping=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": []})

        assert resp.status_code == 400
        assert resp.json()["status"] == "invalid_keys"
        # No mutation: the seeded key survives untouched.
        assert store.is_known("graph1")

    def test_blank_string_entry_returns_400(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1", ""]})

        assert resp.status_code == 400
        assert resp.json()["status"] == "invalid_keys"

    def test_erase_keys_has_require_admin_in_route_table(self):
        """Route-table introspection, in addition to test_require_admin.py's
        _ADMIN_PATHS coverage."""
        from paramem.server.app import app, require_admin

        route_map = {
            route.path: route
            for route in app.routes
            if hasattr(route, "path") and hasattr(route, "dependant")
        }
        route = route_map.get("/debug/erase-keys")
        assert route is not None, "/debug/erase-keys not found in app routes"
        dep_callables = [d.call for d in route.dependant.dependencies]
        assert require_admin in dep_callables, (
            "/debug/erase-keys must carry require_admin dependency"
        )


# ---------------------------------------------------------------------------
# The repair case: registry-known, bookkeeping-absent key
# ---------------------------------------------------------------------------


class TestRepairCase:
    def test_key_without_bookkeeping_is_erased(self, tmp_path, monkeypatch):
        """A key active in the registry but with no bookkeeping record —
        unreachable via /speaker/forget — is erased successfully."""
        store = MemoryStore(replay_enabled=True)
        key = "graph_orphan"
        _seed_key(store, "episodic", key, with_bookkeeping=False)
        # A survivor keeps the tier non-empty so this stays off the reap path.
        _seed_key(store, "episodic", "graph_survivor", with_bookkeeping=True)
        assert store.bookkeeping_for_key(key) is None
        assert store.is_known(key)

        loop = _make_loop_with_store(store)
        cfg = _make_config(tmp_path)
        state = _make_state(tmp_path, loop=loop, config=cfg)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["erased"] == [key]
        assert body["unknown"] == []
        assert not store.is_known(key)
        assert store.bookkeeping_for_key(key) is None


# ---------------------------------------------------------------------------
# Mixed known / unknown partition
# ---------------------------------------------------------------------------


class TestKnownUnknownPartition:
    def test_mixed_request_partitions_correctly(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        known_key = "graph_known"
        survivor_key = "graph_survivor"
        _seed_key(store, "episodic", known_key, with_bookkeeping=True)
        _seed_key(store, "episodic", survivor_key, with_bookkeeping=True)

        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post(
            "/debug/erase-keys",
            json={"keys": [known_key, "graph_never_existed"], "confirm": True},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["erased"] == [known_key]
        assert body["unknown"] == ["graph_never_existed"]
        assert not store.is_known(known_key)
        assert store.is_known(survivor_key)

    def test_all_unknown_keys_erases_nothing(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post(
            "/debug/erase-keys", json={"keys": ["ghost1", "ghost2"], "confirm": True}
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["erased"] == []
        assert body["unknown"] == ["ghost1", "ghost2"]


# ---------------------------------------------------------------------------
# Post-erase bookkeeping: key_metadata.json + promoted_keys
# ---------------------------------------------------------------------------


class TestPostEraseBookkeeping:
    def test_key_metadata_rewritten_and_promoted_keys_cleaned(self, tmp_path, monkeypatch):
        """Uses a real ``ConsolidationLoop`` stub (``object.__new__``) rather
        than a bare ``MagicMock`` — the handler's post-erase call invokes
        ``ConsolidationLoop.write_key_metadata`` directly (the fold's own
        durable writer), and a ``MagicMock``'s version of that method is
        itself a mock that never touches the filesystem (mirrors
        ``tests/server/test_speaker_forget.py::
        test_promoted_keys_and_key_metadata_rewritten``).
        """
        import json

        from paramem.backup.encryption import read_maybe_encrypted
        from paramem.training.consolidation import ConsolidationLoop

        store = MemoryStore(replay_enabled=True)
        key = "graph_promoted"
        survivor_key = "graph_survivor"
        _seed_key(store, "episodic", key, with_bookkeeping=True)
        _seed_key(store, "episodic", survivor_key, with_bookkeeping=True)

        cfg = _make_config(tmp_path)

        # ``_key_metadata_path`` is threaded explicitly from
        # ``cfg.key_metadata_path`` — the same resolution every production
        # construction site performs via ``create_consolidation_loop`` —
        # since ``write_key_metadata`` has no config-fallback of its own.
        loop = object.__new__(ConsolidationLoop)
        loop.store = store
        loop.cycle_count = 0
        loop.model = MagicMock()
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)
        loop.promoted_keys = {key, survivor_key}
        loop._key_metadata_path = cfg.key_metadata_path

        state = _make_state(tmp_path, loop=loop, config=cfg)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": True})

        assert resp.status_code == 200, resp.text
        assert loop.promoted_keys == {survivor_key}

        raw = read_maybe_encrypted(cfg.key_metadata_path)
        written = json.loads(raw.decode("utf-8"))
        assert written["promoted_keys"] == [survivor_key]
        assert key not in written["keys"]
        assert survivor_key in written["keys"]


# ---------------------------------------------------------------------------
# Reap: a tier the erase empties is reaped, not re-stamped
# ---------------------------------------------------------------------------


class TestReap:
    def test_tier_emptied_by_erase_is_reaped(self, tmp_path, monkeypatch):
        # A non-"episodic" tier so the post-reap "episodic" survivor
        # assertion below is not confounded by this erase's own target.
        tier_name = "semantic"
        key = "graph_only_key"

        store = MemoryStore(replay_enabled=True)
        _seed_key(store, tier_name, key, with_bookkeeping=False)

        cfg = _make_config(tmp_path)
        registry_path = cfg.adapter_dir / tier_name / "indexed_key_registry.json"
        store.registry(tier_name).save(registry_path)

        loop = _make_loop_with_store(store)
        loop.model = _make_peft_model("episodic", "semantic", "procedural")
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

        state = _make_state(tmp_path, loop=loop, config=cfg)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["erased"] == [key]
        # Reaped: the tier is gone from the store and the live model no
        # longer carries its adapter.
        assert tier_name not in store.tiers_with_registry()
        assert tier_name not in loop.model.peft_config
        assert "episodic" in loop.model.peft_config  # re-created, never-trained shape

    def test_malformed_tier_name_aborts_before_mutation(self, tmp_path, monkeypatch):
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        tier_name = f"{INTERIM_NAME_PREFIX}foo"  # not a valid stamp
        key = "graph_bad_tier"

        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add(key)

        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": True})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"]["status"] == "malformed_tier_name"
        # Untouched — the abort happens before store.discard_keys runs.
        assert reg.knows(key)


# ---------------------------------------------------------------------------
# Confirmation gate — mirrors POST /interim/discard's Step 0d
# ---------------------------------------------------------------------------


class TestConfirmationGate:
    def test_unconfirmed_returns_owner_ruled_status_without_mutation(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        key = "graph1"
        _seed_key(store, "episodic", key, with_bookkeeping=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key]})

        assert resp.status_code == app_module._DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS, resp.text
        detail = resp.json()["detail"]
        assert detail["error"] == "confirmation_required"
        assert detail["would_erase"] == [key]

        # Nothing mutated.
        assert store.is_known(key)

    def test_confirm_false_explicit_still_refuses(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        key = "graph1"
        _seed_key(store, "episodic", key, with_bookkeeping=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": False})

        assert resp.status_code == app_module._DEBUG_ERASE_KEYS_UNCONFIRMED_STATUS, resp.text
        assert store.is_known(key)


# ---------------------------------------------------------------------------
# Busy guard — mirrors the /speaker/forget guard matrix
# ---------------------------------------------------------------------------


class TestBusyGuard:
    def test_consolidating_refuses_409(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        _seed_key(store, "episodic", "graph1", with_bookkeeping=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop, consolidating=True)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"]})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidating"
        # No mutation: the seeded key survives untouched.
        assert store.is_known("graph1")

    def test_cloud_only_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, loop=None, mode="cloud-only")
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"]})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "cloud_only"

    def test_bg_training_refuses_409(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        loop = _make_loop_with_store(store)
        bg = MagicMock()
        bg.is_training = True
        state = _make_state(tmp_path, loop=loop, background_trainer=bg)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"]})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "training_active"

    def test_base_swap_active_refuses_409(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop, migration={"base_swap_active": True})
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"]})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "base_swap_active"


# ---------------------------------------------------------------------------
# Router reload + response counts
# ---------------------------------------------------------------------------


class TestRouterReloadAndCounts:
    def test_router_reloaded_once_on_mutation(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        key = "graph1"
        _seed_key(store, "episodic", key, with_bookkeeping=True)
        _seed_key(store, "episodic", "graph_survivor", with_bookkeeping=True)

        loop = _make_loop_with_store(store)
        router = MagicMock()
        state = _make_state(tmp_path, loop=loop, router=router)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [key], "confirm": True})

        assert resp.status_code == 200, resp.text
        router.reload.assert_called_once_with()

    def test_response_counts_reflect_post_erase_state(self, tmp_path, monkeypatch):
        store = MemoryStore(replay_enabled=True)
        erased_key = "graph_gone"
        survivor_key = "graph_stays"
        _seed_key(store, "episodic", erased_key, with_bookkeeping=True)
        _seed_key(store, "semantic", survivor_key, with_bookkeeping=True)

        loop = _make_loop_with_store(store)
        state = _make_state(tmp_path, loop=loop)
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": [erased_key], "confirm": True})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 1
        assert body["tiers"] == {"semantic": 1}
        assert body["bookkeeping_total"] == 1
