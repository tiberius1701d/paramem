"""Tests for ``POST /speaker/forget``.

Mocked — no GPU, no real model.  Mirrors the mocking style of
``tests/server/test_gates.py`` and ``tests/server/test_interim_discard.py``.

Coverage
--------
- Guard: the same four ``_consolidation_dispatch_guards`` verdicts as
  ``POST /interim/discard`` (consolidating / bg-training / cloud-only /
  trial-active) refuse with 409 and no store mutation.
- ``store.discard_keys(keys, mode="erase")`` called with exactly the keys
  resolved via ``store.iter_bookkeeping()`` for the speaker: full retirement
  (entry payload, registry — GONE from both active and stale — and
  bookkeeping, dropped in lockstep via delegation to ``MemoryStore.delete``);
  registry saved for affected tiers; each affected tier's on-disk
  ``graph.json`` has the erased keys' edges surgically removed
  (``persistence.erase_keys_from_graph_file``); ``loop.promoted_keys`` and
  on-disk ``key_metadata.json`` no longer carry the erased keys.
- A tier reduced to zero known keys by the erase is reaped instead of
  manifest re-stamped: live PEFT unmount (``detach_adapters``) plus on-disk
  artifact removal (``reap_tier_artifacts``), for both interim and main
  tiers, always leaving a serviceable model (``ensure_adapters`` + episodic
  switch + ``_state["model"]`` re-publish).  A surviving tier keeps today's
  registry-level posture (re-stamp, not reaped).
- Ring-lifecycle incidents resolved only when this forget's reap empties the
  interim ring entirely; untouched when a sibling interim tier survives.
- Speaker profile removed: ``speaker_store.remove`` called; response reflects
  the bool return.
- Pending sessions for the speaker discarded: ``discard_sessions`` called;
  response lists them.
- Router reloaded exactly once after the mutation.
- ``_state["consolidating"]`` is ``True`` for the duration of the mutation
  and cleared in a ``finally`` regardless of outcome (including a reap
  failure, which propagates as a 500).
- Auth: ``/speaker/forget`` carries the ``require_admin`` dependency in the
  real app route table.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
from fastapi.testclient import TestClient
from peft import PeftModel

import paramem.server.app as app_module
from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    TokenizerFingerprint,
    find_live_slot,
    read_manifest,
    write_manifest,
)
from paramem.memory.entry import compute_simhash
from paramem.memory.persistence import (
    _IK_KEY_ATTR,
    iter_entries,
    load_memory_from_disk,
    save_memory_to_disk,
)
from paramem.memory.store import MemoryStore
from paramem.server.incidents import read_incidents, record_incident
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> MagicMock:
    """Minimal config mock with adapter_dir and key_metadata_path under tmp_path.

    ``key_metadata_path`` must be a real ``Path`` (not a bare MagicMock
    attribute) — tests that use a REAL ``ConsolidationLoop`` stub (rather
    than the ``MagicMock`` loops built by :func:`_make_loop_with_store` /
    :func:`_make_loop`) thread it into ``loop._key_metadata_path`` so
    ``loop.write_key_metadata()`` has somewhere to write.
    """
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.key_metadata_path = tmp_path / "registry" / "key_metadata.json"
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _prep_loop_promoted_keys(loop: MagicMock) -> None:
    """Set ``loop.promoted_keys`` to a real ``set`` on a mock loop.

    The handler calls ``loop.promoted_keys.difference_update(erased_keys)``
    directly (independent of the key-metadata write), so this must be a real
    ``set`` rather than an auto-generated ``MagicMock`` attribute.  The
    subsequent ``loop.write_key_metadata()`` call is itself a mock on these
    ``MagicMock`` loops and never touches the filesystem, so no destination
    wiring is needed here.
    """
    loop.promoted_keys = set()


def _make_peft_model(*adapter_names: str) -> MagicMock:
    """Stub PeftModel — mirrors ``_make_peft_model`` in
    ``tests/server/test_interim_discard.py``: a real dict-backed
    ``peft_config`` so membership/iteration/deletion behave like PEFT.
    """
    model = MagicMock(spec=PeftModel)
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _make_loop_with_store(store: MemoryStore) -> MagicMock:
    """Build a MagicMock ConsolidationLoop wrapping a real MemoryStore.

    Used by tests that need real store mutation semantics (registry, entry,
    bookkeeping, simhash) but drive everything else on the loop through a
    mock.  Pre-wires ``loop.promoted_keys`` as a real ``set`` (see
    :func:`_prep_loop_promoted_keys`) so the handler's post-erase
    ``loop.promoted_keys.difference_update(...)`` call does not crash on mock
    plumbing unrelated to the behaviour under test; the subsequent
    ``loop.write_key_metadata()`` call is itself a mock on this ``MagicMock``
    loop.

    Also wires ``loop.model`` (a bare, non-PEFT ``MagicMock`` by default —
    ``detach_adapters`` no-ops on it) and ``loop.ensure_adapters`` (returns
    whatever ``loop.model`` currently is, read dynamically at call time) so
    the reap path (``_reap_emptied_tiers``) has something to call without
    crashing on tests that never exercise a reap.  Tests exercising an
    actual reap replace ``loop.model`` with a :func:`_make_peft_model` stub
    before posting.
    """
    loop = MagicMock()
    loop.store = store
    _prep_loop_promoted_keys(loop)
    loop.model = MagicMock()
    loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)
    return loop


def _make_loop(speaker_id: str, keys: list[str]) -> MagicMock:
    """Build a MagicMock ConsolidationLoop; store.iter_bookkeeping yields *keys* for *speaker_id*.

    The store exposes two main tiers (``episodic`` and ``semantic``) whose
    ``KeyRegistry`` objects contain the supplied keys so registry saves can be
    verified.  The simhash dicts include the keys so the simhash clean-up branch
    is exercised.

    ``/forget`` routes through ``store.iter_bookkeeping()`` to resolve speaker
    keys (``merger.graph`` is cleared at cycle-end by the cycle's finally-block
    reset, so the old graph-based ``keys_for_speaker`` path is unavailable).
    It then calls
    ``store.discard_keys(keys, mode="erase")`` (the shared helper).  Tests verify
    that the helper is called with the correct arguments.

    Also pre-wires what the handler's post-erase
    ``loop.promoted_keys.difference_update(...)`` call needs (see
    :func:`_prep_loop_promoted_keys`); the subsequent
    ``loop.write_key_metadata()`` call is itself a mock on this ``MagicMock``
    loop.
    """
    loop = MagicMock()
    _prep_loop_promoted_keys(loop)
    loop.model = MagicMock()
    loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

    # iter_bookkeeping returns bookkeeping records keyed by speaker_id.
    # The handler iterates all records and filters by record.get("speaker_id").
    bk_records = [(k, {"speaker_id": speaker_id, "relation_type": "episodic"}) for k in keys]
    loop.store.iter_bookkeeping.return_value = iter(bk_records)

    # Per-tier KeyRegistry mocks.
    ep_registry = MagicMock(spec=KeyRegistry)
    ep_registry.__contains__ = MagicMock(side_effect=lambda k: k in keys)
    ep_registry.is_stale = MagicMock(return_value=False)
    ep_registry.knows = MagicMock(side_effect=lambda k: k in keys)
    # save_bytes must return valid bytes so the handler can compute sha256 for
    # the post-erase re-stamp. Distinct pre/post values are not needed here
    # because discard_keys is mocked (it does not mutate the mock registry);
    # any stable bytes value is sufficient.
    ep_registry.save_bytes.return_value = (
        b'{"active_keys": [], "fidelity_history": {}, "stale": {}, "simhash": {}}'
    )
    sem_registry = MagicMock(spec=KeyRegistry)
    sem_registry.__contains__ = MagicMock(side_effect=lambda _: False)
    sem_registry.is_stale = MagicMock(return_value=False)
    sem_registry.knows = MagicMock(side_effect=lambda _: False)

    # Store: tiers_with_registry returns episodic + semantic.
    loop.store.tiers_with_registry.return_value = ["episodic", "semantic"]
    loop.store.registry.side_effect = lambda t: ep_registry if t == "episodic" else sem_registry

    loop._ep_registry = ep_registry
    loop._sem_registry = sem_registry
    return loop


def _make_speaker_store(speaker_id: str, *, returns: bool = True) -> MagicMock:
    """SpeakerStore mock whose remove(speaker_id) returns *returns*."""
    store = MagicMock()
    store.remove.return_value = returns
    return store


def _make_buffer(speaker_id: str, conv_ids: list[str]) -> MagicMock:
    """SessionBuffer mock with _sessions carrying *conv_ids* attributed to *speaker_id*."""
    buf = MagicMock()
    buf._sessions = {cid: {"speaker_id": speaker_id, "speaker": "Test User"} for cid in conv_ids}
    return buf


def _make_state(
    tmp_path: Path,
    *,
    loop=None,
    speaker_store=None,
    buffer=None,
    config=None,
    mode: str = "local",
    consolidating: bool = False,
    background_trainer=None,
    migration=None,
    router=None,
    adapter_manifest_status=None,
) -> dict:
    """Build a minimal _state dict for endpoint tests.

    Extended (2026-08-05) with the keys the guard/reap rework reads:
    ``mode``/``consolidating``/``background_trainer``/``migration`` feed
    ``_consolidation_dispatch_guards`` (plain ``dict[...]`` access, so every
    caller of this helper must set them — the guard would otherwise
    ``KeyError``); ``router`` is reloaded in the handler's no-await tail;
    ``adapter_manifest_status`` is popped per reaped tier by
    ``_reap_emptied_tiers``.  ``model``/``tokenizer``/``memory_store`` mirror
    what the real lifespan publishes — ``memory_store`` is the SAME object as
    ``loop.store`` so a caller inspecting either sees identical state, and
    ``model`` is the same object as ``loop.model`` so a reap's
    ``_state["model"]`` re-publish is observable via either handle.
    """
    return {
        "config": config or _make_config(tmp_path),
        "consolidation_loop": loop,
        "speaker_store": speaker_store,
        "session_buffer": buffer or MagicMock(),
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


# ---------------------------------------------------------------------------
# Happy-path: mark_stale removes keys from KeyRegistry and persists
# ---------------------------------------------------------------------------


class TestMarkStaleKeys:
    """Keys from store.iter_bookkeeping are hard-erased via store.discard_keys(mode="erase")."""

    def test_keys_removed_from_registry_and_saved(self, tmp_path, monkeypatch):
        """store.discard_keys called with mode='erase'; registry.save called for affected tier."""
        speaker_id = "speaker0"
        keys = ["graph1", "graph3"]
        loop = _make_loop(speaker_id, keys)
        cfg = _make_config(tmp_path)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        with patch("paramem.memory.persistence.save_registry"):
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Response lists stale keys (sorted).
        assert sorted(body["erased_keys"]) == sorted(keys)

        # iter_bookkeeping was called (not keys_for_speaker which used merger.graph).
        loop.store.iter_bookkeeping.assert_called_once_with()

        # store.discard_keys must be called with mode="erase" — full retirement
        # (entries + registry/simhash + bookkeeping dropped in lockstep, via
        # delegation to MemoryStore.delete per key), not soft-stale.  Verify the
        # helper was called; the erase contract itself is pinned in
        # tests/test_memory_store.py::TestDiscardKeys.
        loop.store.discard_keys.assert_called_once_with(sorted(keys), mode="erase")

        # registry.save called with the episodic adapter path (episodic contains the keys).
        ep_reg = loop._ep_registry
        expected_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        ep_reg.save.assert_called_once_with(expected_path)

        # Semantic registry — no keys → save not called.
        loop._sem_registry.save.assert_not_called()

    def test_simhash_entries_removed_from_store(self, tmp_path, monkeypatch):
        """Erased key is absent from the in-memory simhash map after erase.

        Uses a real MemoryStore so discard_keys(mode='erase') actually removes
        the simhash entry.  Simhash is now unified in indexed_key_registry.json
        (not a separate sidecar), so we verify the in-memory fingerprint map
        rather than a separate simhash_registry.json save.
        """
        speaker_id = "speaker0"
        keys = ["graph2"]
        key = keys[0]

        # Build a real MemoryStore seeded with the key in episodic.
        real_store = MemoryStore(replay_enabled=True)
        fingerprint = compute_simhash(key, "Alice", "lives_in", "Berlin")
        real_store.put(
            "episodic",
            key,
            {"key": key, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=fingerprint,
        )
        # seed bookkeeping so iter_bookkeeping() resolves the key for this speaker.
        real_store.set_bookkeeping(
            key,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # Assemble the loop mock, replacing .store with the real MemoryStore.
        loop = _make_loop(speaker_id, keys)
        loop.store = real_store

        cfg = _make_config(tmp_path)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        # The real discard_keys(mode="erase") ran — key must be absent from
        # the in-memory simhash map (active AND stale — a hard erase removes both).
        assert key not in real_store.tier_simhashes("episodic", include_stale=True), (
            f"Key {key!r} still present in episodic simhash after erase"
        )

    def test_no_keys_no_registry_mutations(self, tmp_path, monkeypatch):
        """When iter_bookkeeping yields no keys for the speaker, discard_keys is not called."""
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        with patch("paramem.memory.persistence.save_registry") as mock_save:
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["erased_keys"] == []
        # No registry saves because there are no keys to process.
        mock_save.assert_not_called()
        loop.store.discard_keys.assert_not_called()


# ---------------------------------------------------------------------------
# Speaker profile removal
# ---------------------------------------------------------------------------


class TestSpeakerProfileRemoval:
    """speaker_store.remove called; response removed_speaker reflects its bool."""

    def test_profile_found_and_removed(self, tmp_path, monkeypatch):
        """SpeakerStore.remove returns True → removed_speaker is True."""
        speaker_id = "speaker1"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id, returns=True),
            buffer=_make_buffer(speaker_id, []),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed_speaker"] is True
        state["speaker_store"].remove.assert_called_once_with(speaker_id)

    def test_profile_not_found(self, tmp_path, monkeypatch):
        """SpeakerStore.remove returns False → removed_speaker is False."""
        speaker_id = "speaker99"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id, returns=False),
            buffer=_make_buffer(speaker_id, []),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed_speaker"] is False

    def test_no_speaker_store(self, tmp_path, monkeypatch):
        """When speaker_store is None (voice disabled), removed_speaker is False."""
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=None,
            buffer=_make_buffer(speaker_id, []),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed_speaker"] is False


# ---------------------------------------------------------------------------
# Pending session discard
# ---------------------------------------------------------------------------


class TestPendingSessionDiscard:
    """discard_sessions called with the speaker's conversation IDs; response lists them."""

    def test_pending_sessions_discarded(self, tmp_path, monkeypatch):
        """Sessions attributed to the speaker are discarded; ids in response."""
        speaker_id = "speaker0"
        conv_ids = ["conv-a", "conv-b"]
        loop = _make_loop(speaker_id, [])
        buffer = _make_buffer(speaker_id, conv_ids)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=buffer,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert sorted(body["discarded_sessions"]) == sorted(conv_ids)
        buffer.discard_sessions.assert_called_once_with(conv_ids)

    def test_no_pending_sessions(self, tmp_path, monkeypatch):
        """When no pending sessions exist for the speaker, discarded_sessions is empty."""
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, [])
        buffer = MagicMock()
        buffer._sessions = {"other-conv": {"speaker_id": "speaker1"}}
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=buffer,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["discarded_sessions"] == []
        buffer.discard_sessions.assert_not_called()

    def test_sessions_with_mixed_speakers_only_discards_target(self, tmp_path, monkeypatch):
        """Only sessions whose speaker_id matches the request are discarded."""
        speaker_id = "speaker0"
        other_id = "speaker1"
        loop = _make_loop(speaker_id, [])
        buffer = MagicMock()
        buffer._sessions = {
            "conv-target": {"speaker_id": speaker_id},
            "conv-other": {"speaker_id": other_id},
            "conv-unidentified": {},
        }
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=buffer,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["discarded_sessions"] == ["conv-target"]
        buffer.discard_sessions.assert_called_once_with(["conv-target"])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    """Cloud-only mode (no local model / consolidation loop) refuses with 409."""

    def test_cloud_only_refuses_409(self, tmp_path, monkeypatch):
        """When the server is cloud-only, the guard refuses before the loop
        is ever touched — inverts the old "loop is None -> 503" behaviour:
        the loop is now get-or-created lazily, so absence alone is no longer
        an error condition; only the guard's cloud_only verdict is."""
        state = _make_state(
            tmp_path,
            loop=None,
            speaker_store=_make_speaker_store("speaker0"),
            buffer=MagicMock(),
            mode="cloud-only",
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": "speaker0"})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "cloud_only"

    def test_legacy_strategy_field_is_ignored_not_rejected(self, tmp_path, monkeypatch):
        """A caller still sending a retired ``strategy`` field is unaffected: the
        request schema's default ``extra="ignore"`` (pydantic) drops the field
        silently and the erase proceeds normally rather than 400ing."""
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post(
            "/speaker/forget",
            json={"speaker_id": speaker_id, "strategy": "mark_stale"},
        )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth: require_admin dependency present in route table
# ---------------------------------------------------------------------------


class TestAuthDependency:
    """/speaker/forget carries require_admin in the real app route table."""

    def test_speaker_forget_has_require_admin(self):
        """Route table introspection: require_admin must be in /speaker/forget dependencies."""
        from paramem.server.app import app, require_admin

        route_map = {
            route.path: route
            for route in app.routes
            if hasattr(route, "path") and hasattr(route, "dependant")
        }
        route = route_map.get("/speaker/forget")
        assert route is not None, "/speaker/forget not found in app routes"
        dep_callables = [d.call for d in route.dependant.dependencies]
        assert require_admin in dep_callables, "/speaker/forget must carry require_admin dependency"


# ---------------------------------------------------------------------------
# Live slot manifest re-stamp after registry erase
# ---------------------------------------------------------------------------


def _minimal_manifest_for_tier(name: str, registry_sha256: str, key_count: int) -> AdapterManifest:
    """Build a minimal AdapterManifest for use in re-stamp integration tests."""
    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        trained_at="2026-06-12T00:00:00Z",
        base_model=BaseModelFingerprint(repo="hf/model", sha="abc123", hash="sha256:deadbeef"),
        tokenizer=TokenizerFingerprint(
            name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
        ),
        lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
        registry_sha256=registry_sha256,
        key_count=key_count,
    )


class TestLiveSlotManifestReStamp:
    """After /speaker/forget erases a key, find_live_slot must rebind to the new registry hash.

    Root cause being tested: out-of-fold registry mutations change the registry's
    SHA-256 but previously left the live slot's meta.json unchanged → on restart
    find_live_slot (exact hash match) returned None → preload 0/N → tier dead.

    The fix (inline in the /speaker/forget handler) must:
    (a) Re-stamp meta.json with the POST-erase registry hash.
    (b) find_live_slot(kind_dir, H_new) returns the slot.
    (c) find_live_slot(kind_dir, H_old) returns None (old hash obsolete).
    (d) manifest.key_count is unchanged (counts WEIGHT keys, not registry keys).
    """

    def test_live_slot_rebound_to_new_hash_after_forget(self, tmp_path, monkeypatch):
        """After /speaker/forget erases a key, find_live_slot rebinds to the new hash."""
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        # Build a real KeyRegistry with both keys in the episodic tier.
        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)

        # Build the loop mock, using the real MemoryStore for registry + discard_keys.
        loop = _make_loop_with_store(real_store)

        # Add a simhash entry so the simhash-clean branch is exercised too.
        # IMPORTANT: this must happen BEFORE computing H_old so that save_bytes()
        # (which now includes the unified simhash map) yields a hash that matches
        # what app.py computes as the pre-erase hash at forget time.
        fingerprint = compute_simhash(key_to_forget, "Alice", "lives_in", "Berlin")
        real_store.put(
            tier_name,
            key_to_forget,
            {
                "key": key_to_forget,
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
            },
            simhash=fingerprint,
        )

        # seed bookkeeping so iter_bookkeeping() resolves key_to_forget for speaker_id.
        # Bookkeeping is separate from save_bytes() (KeyRegistry only), so this
        # does NOT affect H_old — the computation below remains correct.
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # Compute H_old — the hash of the registry BEFORE the erase.
        # Must be computed after all registry mutations (including simhash writes)
        # because save_bytes() now includes the unified simhash map.
        h_old = hashlib.sha256(ep_reg.save_bytes()).hexdigest()

        # Persist the pre-erase registry to disk. The handler reads the
        # PRE-ERASE hash from the ON-DISK file via tier_registry_sha256
        # (not by re-serialising the in-memory registry — see the root-cause
        # comment at the /speaker/forget call site), so the on-disk bytes must
        # exist and match H_old for find_live_slot to locate the manifest below.
        registry_path = tmp_path / "adapters" / tier_name / "indexed_key_registry.json"
        ep_reg.save(registry_path)

        # Write a real slot directory with a manifest stamped with H_old.
        # key_count reflects both keys (counts weight keys, not registry keys).
        slot_dir = tmp_path / "adapters" / tier_name / "20260612-000000"
        slot_dir.mkdir(parents=True)
        original_manifest = _minimal_manifest_for_tier(tier_name, h_old, key_count=2)
        write_manifest(slot_dir, original_manifest)

        # Sanity: the pre-erase manifest is locatable.
        kind_dir = tmp_path / "adapters" / tier_name
        assert find_live_slot(kind_dir, h_old) == slot_dir

        # Build config pointing at tmp_path/adapters.
        cfg = _make_config(tmp_path)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        with patch("paramem.memory.persistence.save_registry"):
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        # Compute H_new — the hash of the mutated registry after the erase.
        # (The real discard_keys ran, so ep_reg no longer contains key_to_forget.)
        ep_reg_after = real_store.registry(tier_name)
        h_new = hashlib.sha256(ep_reg_after.save_bytes()).hexdigest()

        # (i) find_live_slot rebinds to the NEW hash.
        assert find_live_slot(kind_dir, h_new) == slot_dir, (
            "find_live_slot must return the slot after re-stamp with H_new"
        )

        # (ii) The OLD hash no longer matches.
        assert find_live_slot(kind_dir, h_old) is None, (
            "find_live_slot must return None for the pre-erase hash after re-stamp"
        )

        # (iii) key_count in the manifest is unchanged (counts weight keys, not registry keys).
        manifest_after = read_manifest(slot_dir)
        assert manifest_after.key_count == original_manifest.key_count, (
            "key_count must remain unchanged — it counts keys in the adapter weights, "
            "which /forget does not modify"
        )

    def test_rebind_survives_registry_payload_shape_change(self, tmp_path, monkeypatch):
        """Cross-version regression: the pre-erase hash must come from the ON-DISK
        bytes, not from re-serialising the in-memory registry.

        Models the deploy -> first-fold window: the registry file on disk was
        last written by an OLDER build whose ``KeyRegistry.save_bytes()`` still
        emitted a 5-key payload (including a now-retired ``"health"`` field);
        the manifest's ``registry_sha256`` was stamped against those exact
        bytes. The server has since been upgraded — the resident in-memory
        ``KeyRegistry`` (loaded from the same file, which tolerantly ignores
        the unknown ``"health"`` key) now serialises to a DIFFERENT 4-key
        payload if re-hashed in memory. A /speaker/forget erase must still
        locate and re-stamp the live slot by reading the literal on-disk
        bytes — proven by asserting the two hashes diverge before running the
        request, so this test cannot pass by accident.
        """
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # Literal LEGACY on-disk payload — the exact 5-key shape a pre-migration
        # build's KeyRegistry.save_bytes() emitted, including the retired
        # "health" field. Intentionally NOT produced via ep_reg.save_bytes():
        # this is the file a prior server version actually wrote to disk, and
        # the CURRENT in-memory registry never re-derives it.
        legacy_registry_bytes = (
            b'{"active_keys": ["graph1", "graph2"], "fidelity_history": {}, '
            b'"health": null, "stale": {}, "simhash": {}}'
        )
        registry_path = tmp_path / "adapters" / tier_name / "indexed_key_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_bytes(legacy_registry_bytes)
        h_legacy = hashlib.sha256(legacy_registry_bytes).hexdigest()

        # Sanity: the CURRENT in-memory serialisation diverges from the legacy
        # on-disk bytes — if it didn't, this test couldn't distinguish the
        # root-cause fix from the bug it replaces.
        assert hashlib.sha256(ep_reg.save_bytes()).hexdigest() != h_legacy, (
            "test setup error: in-memory and legacy on-disk hashes must differ"
        )

        slot_dir = tmp_path / "adapters" / tier_name / "20260612-000000"
        slot_dir.mkdir(parents=True)
        original_manifest = _minimal_manifest_for_tier(tier_name, h_legacy, key_count=2)
        write_manifest(slot_dir, original_manifest)

        kind_dir = tmp_path / "adapters" / tier_name
        assert find_live_slot(kind_dir, h_legacy) == slot_dir

        loop = _make_loop_with_store(real_store)

        cfg = _make_config(tmp_path)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        with patch("paramem.memory.persistence.save_registry"):
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        ep_reg_after = real_store.registry(tier_name)
        h_new = hashlib.sha256(ep_reg_after.save_bytes()).hexdigest()
        assert find_live_slot(kind_dir, h_new) == slot_dir, (
            "rebind must succeed by reading the on-disk pre-erase bytes even "
            "though the current in-memory serialiser can no longer reproduce "
            "the legacy payload shape they were hashed from"
        )
        manifest_after = read_manifest(slot_dir)
        assert manifest_after.registry_sha256 == h_new
        assert manifest_after.key_count == original_manifest.key_count

    def test_no_slot_logs_error_without_failing(self, tmp_path, monkeypatch, caplog):
        """When no slot matches the pre-erase hash, an ERROR is logged but the request succeeds.

        This covers the already-orphaned slot case (e.g. after a prior crash between
        registry.save and manifest re-stamp).  The /forget must not fail — partial
        re-stamp state is recoverable via a full consolidation fold.
        """
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        # Build a real KeyRegistry with the key.  A survivor key keeps the
        # tier non-empty after the erase so this test stays on the re-stamp
        # branch under test rather than falling into the reap branch.
        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)

        # seed bookkeeping so iter_bookkeeping() resolves key_to_forget for speaker_id.
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # Persist the pre-erase registry to disk so the handler's pre-erase hash
        # (tier_registry_sha256, read from the on-disk file) is non-empty — this
        # models the already-orphaned-slot case (a prior crash between
        # registry.save and manifest re-stamp), not an absent registry.
        registry_path = tmp_path / "adapters" / tier_name / "indexed_key_registry.json"
        ep_reg.save(registry_path)

        # Write a slot with a DIFFERENT registry_sha256 so find_live_slot won't match H_old.
        slot_dir = tmp_path / "adapters" / tier_name / "20260612-000000"
        slot_dir.mkdir(parents=True)
        mismatched_manifest = _minimal_manifest_for_tier(tier_name, "deadbeef" * 8, key_count=1)
        write_manifest(slot_dir, mismatched_manifest)

        cfg = _make_config(tmp_path)

        loop = _make_loop_with_store(real_store)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        with patch("paramem.memory.persistence.save_registry"):
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        # An ERROR must be logged about the missing slot.
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("slot already orphaned" in msg for msg in error_messages), (
            f"Expected orphaned-slot ERROR in logs, got: {error_messages}"
        )

    def test_empty_pre_hash_skips_restamp_and_find_live_slot(self, tmp_path, monkeypatch, caplog):
        """Tier has in-memory keys but no on-disk registry (pre-erase hash == "").

        Must: return 200, log a WARNING (not ERROR), never consult
        find_live_slot for this tier, and leave any existing manifest
        untouched — an unreadable/absent registry must not bind a stray
        ""-stamped slot.
        """
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        # A survivor key keeps the tier non-empty after the erase so this
        # test stays on the re-stamp branch under test.
        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )
        # Deliberately NOT saved to disk — no indexed_key_registry.json
        # exists, so tier_registry_sha256 (the pre-erase hash) is "".

        cfg = _make_config(tmp_path)

        # A stray slot dir with a manifest that must be left byte-for-byte
        # untouched — proves the skip branch never re-stamps anything for
        # this tier (its registry_sha256 is irrelevant here: find_live_slot
        # is never even consulted for a "" pre-erase hash).
        slot_dir = cfg.adapter_dir / tier_name / "20260101-000000"
        slot_dir.mkdir(parents=True)
        stray_manifest = _minimal_manifest_for_tier(tier_name, "unrelated" * 8, key_count=0)
        write_manifest(slot_dir, stray_manifest)

        loop = _make_loop_with_store(real_store)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        caplog.set_level(logging.WARNING, logger="paramem.server.app")
        with patch("paramem.adapters.manifest.find_live_slot") as mock_find_live_slot:
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        mock_find_live_slot.assert_not_called()

        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "had no readable pre-erase registry on disk" in msg for msg in warning_messages
        ), f"Expected empty-pre-hash WARNING in logs, got: {warning_messages}"

        # No manifest touched.
        assert read_manifest(slot_dir) == stray_manifest

    def test_simulate_venue_no_weight_slot_no_error_log(self, tmp_path, monkeypatch, caplog):
        """Simulate-mode forget: the tier registry is real and present on
        disk (non-empty pre-erase hash) but no weight-slot manifest exists
        at all — the simulate venue never mints one.  find_live_slot would
        always return None here, which is NOT "the slot is orphaned"; it is
        "there is no slot to re-stamp".  Must not log an ERROR."""
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        # A survivor key keeps the tier non-empty after the erase so this
        # test stays on the re-stamp branch under test.
        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # A real, non-empty pre-erase registry on disk — the simulate venue
        # writes indexed_key_registry.json same as train — but NO weight-slot
        # subdirectory (no meta.json) anywhere under the tier root.
        cfg = _make_config(tmp_path)
        registry_path = cfg.adapter_dir / tier_name / "indexed_key_registry.json"
        ep_reg.save(registry_path)

        loop = _make_loop_with_store(real_store)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        caplog.set_level(logging.DEBUG, logger="paramem.server.app")
        with patch("paramem.adapters.manifest.find_live_slot") as mock_find_live_slot:
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        mock_find_live_slot.assert_not_called()

        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages == [], (
            f"simulate-venue forget must not log an ERROR; got: {error_messages}"
        )


# ---------------------------------------------------------------------------
# Interim-tier arc: registry/manifest paths must use the nested interim
# layout (adapter_dir/episodic/interim_<stamp>/), not a flat
# adapter_dir/<tier_name>/ join.
# ---------------------------------------------------------------------------


class TestInterimTierForget:
    """/speaker/forget on a speaker whose key lives in an interim tier.

    Root cause being tested: a flat ``adapter_dir / tier_name`` join is wrong
    for interim tier names (``episodic_interim_<stamp>``) — the real root is
    ``adapter_dir/episodic/interim_<stamp>/`` (resolved by
    :func:`~paramem.memory.interim_adapter.adapter_slot_root_for_name`).  The
    flat join would mint a stray top-level ``adapter_dir/episodic_interim_<stamp>/``
    directory (which bricks the next boot via ``detect_legacy_adapter_layout``),
    never rewrite the interim slot's real registry (the erase reverts on
    reload), and read the pre-erase hash as ``""`` (the manifest re-stamp
    then silently fails to rebind the slot).
    """

    def test_interim_key_erased_durably_and_manifest_restamped(self, tmp_path, monkeypatch):
        """Erasing an interim-tier key persists to the nested interim root and re-stamps."""
        from paramem.memory.interim_adapter import (
            INTERIM_NAME_PREFIX,
            detect_legacy_adapter_layout,
        )

        stamp = "20260803T1200"
        tier_name = f"{INTERIM_NAME_PREFIX}{stamp}"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)

        # Build a real MemoryStore with a real interim-tier KeyRegistry.
        real_store = MemoryStore(replay_enabled=True)
        interim_reg = real_store.registry(tier_name)
        interim_reg.add(key_to_forget)
        interim_reg.add(key_to_keep)

        fingerprint = compute_simhash(key_to_forget, "Alice", "lives_in", "Berlin")
        real_store.put(
            tier_name,
            key_to_forget,
            {"key": key_to_forget, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=fingerprint,
        )
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="episodic",
            first_seen="",
        )
        # key_to_keep belongs to a different speaker — a deliberate survivor
        # (not an un-bookkept orphan) so the tier is genuinely non-empty
        # after the erase and this test stays on the re-stamp branch; the
        # reap branch has its own sibling test right below.
        real_store.set_bookkeeping(
            key_to_keep,
            speaker_id="speaker1",
            relation_type="episodic",
            first_seen="",
        )

        # The REAL nested interim root: adapter_dir/episodic/interim_<stamp>/.
        interim_root = adapter_dir / "episodic" / f"interim_{stamp}"
        h_old = hashlib.sha256(interim_reg.save_bytes()).hexdigest()
        registry_path = interim_root / "indexed_key_registry.json"
        interim_reg.save(registry_path)

        # Weight slot subdir, stamped with the pre-erase hash.
        slot_dir = interim_root / "20260803-120000"
        slot_dir.mkdir(parents=True)
        original_manifest = _minimal_manifest_for_tier(tier_name, h_old, key_count=2)
        write_manifest(slot_dir, original_manifest)

        cfg = _make_config(tmp_path)
        assert cfg.adapter_dir == adapter_dir

        loop = _make_loop_with_store(real_store)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        # (a) No legacy top-level episodic_interim_* dir was minted, and none
        # of the ordinary paths (KeyRegistry.save's mkdir(parents=True)) can
        # have created one either.
        assert detect_legacy_adapter_layout(adapter_dir) == []
        assert not (adapter_dir / tier_name).exists()

        # (b) Durable erase: a FRESH on-disk walk (independent of the live
        # in-memory store) shows the key AND its simhash gone from the real
        # interim root; the sibling key survives.
        fresh_registries = MemoryStore.read_registries_from_disk(adapter_dir)
        assert tier_name in fresh_registries, (
            f"interim tier {tier_name!r} not found on disk at {interim_root} "
            f"(registries found: {sorted(fresh_registries)})"
        )
        fresh_interim_reg = fresh_registries[tier_name]
        assert not fresh_interim_reg.knows(key_to_forget)
        assert not fresh_interim_reg.has_simhash(key_to_forget)
        assert fresh_interim_reg.knows(key_to_keep)

        # (c) The slot's meta.json registry_sha256 is re-stamped to the
        # post-erase hash — i.e. it matches the literal on-disk bytes.
        h_new = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        manifest_after = read_manifest(slot_dir)
        assert manifest_after.registry_sha256 == h_new
        assert find_live_slot(interim_root, h_new) == slot_dir
        assert find_live_slot(interim_root, h_old) is None

    def test_interim_tier_fully_emptied_is_reaped_not_restamped(self, tmp_path, monkeypatch):
        """Sibling of the test above: no survivor key this time, so the
        nested interim root is reaped wholesale instead of re-stamped — the
        on-disk dir is gone, not left with a re-stamped (but now-erased)
        registry."""
        from paramem.memory.interim_adapter import (
            INTERIM_NAME_PREFIX,
            detect_legacy_adapter_layout,
        )

        stamp = "20260803T1300"
        tier_name = f"{INTERIM_NAME_PREFIX}{stamp}"
        key_to_forget = "graph1"
        speaker_id = "speaker0"

        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)

        real_store = MemoryStore(replay_enabled=True)
        interim_reg = real_store.registry(tier_name)
        interim_reg.add(key_to_forget)
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="episodic",
            first_seen="",
        )

        interim_root = adapter_dir / "episodic" / f"interim_{stamp}"
        interim_reg.save(interim_root / "indexed_key_registry.json")
        slot_dir = interim_root / "20260803-130000"
        slot_dir.mkdir(parents=True)
        write_manifest(
            slot_dir,
            _minimal_manifest_for_tier(
                tier_name, hashlib.sha256(interim_reg.save_bytes()).hexdigest(), key_count=1
            ),
        )

        cfg = _make_config(tmp_path)
        assert cfg.adapter_dir == adapter_dir

        model = _make_peft_model("episodic", "semantic", "procedural", tier_name)
        loop = _make_loop_with_store(real_store)
        loop.model = model
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reaped_tiers"] == [tier_name]
        assert body["unloaded_adapters"] == [tier_name]
        assert body["removed_dirs"] == [f"interim_{stamp}"]

        # No legacy top-level dir was minted either.
        assert detect_legacy_adapter_layout(adapter_dir) == []
        assert not (adapter_dir / tier_name).exists()

        # The whole nested interim root is gone — not left behind re-stamped.
        assert not interim_root.exists()
        assert tier_name not in real_store.tiers_with_registry()
        assert tier_name not in model.peft_config
        assert "episodic" in model.peft_config

    def test_malformed_interim_tier_name_aborts_before_mutation(self, tmp_path, monkeypatch):
        """A malformed interim tier name errors BEFORE the store is mutated."""
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        # "foo" does not parse as an INTERIM_STAMP_FORMAT stamp — a stray
        # interim tier name the store should never carry, but which
        # adapter_slot_root_for_name must reject rather than silently
        # building a bogus path from it.
        tier_name = f"{INTERIM_NAME_PREFIX}foo"
        key_to_forget = "graph1"
        speaker_id = "speaker0"

        real_store = MemoryStore(replay_enabled=True)
        malformed_reg = real_store.registry(tier_name)
        malformed_reg.add(key_to_forget)
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="episodic",
            first_seen="",
        )

        cfg = _make_config(tmp_path)

        loop = _make_loop_with_store(real_store)

        speaker_store = _make_speaker_store(speaker_id)
        buffer = _make_buffer(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=speaker_store,
            buffer=buffer,
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 500, resp.text

        # The store must be untouched — the abort happens before
        # store.discard_keys runs.
        assert malformed_reg.knows(key_to_forget)

        # Nothing downstream of the erase ran either — the abort happens
        # before ANY mutation, not just before the registry erase.
        speaker_store.remove.assert_not_called()
        buffer.discard_sessions.assert_not_called()


# ---------------------------------------------------------------------------
# Full retirement: repeat-forget idempotency, entry/bookkeeping/router state,
# and promoted_keys + key_metadata.json rewrite
# ---------------------------------------------------------------------------


class TestFullRetirement:
    """The erased key is gone from every RAM structure and from
    ``key_metadata.json`` — not just the registry pointer."""

    def test_repeat_forget_is_idempotent(self, tmp_path, monkeypatch):
        """A second forget for the same speaker reports nothing left to erase."""
        speaker_id = "speaker0"
        key = "graph1"

        real_store = MemoryStore(replay_enabled=True)
        real_store.put(
            "episodic",
            key,
            {"key": key, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=compute_simhash(key, "Alice", "lives_in", "Berlin"),
        )
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        loop = _make_loop_with_store(real_store)
        cfg = _make_config(tmp_path)
        # A real SpeakerStore.remove returns True only the first time (the
        # profile is gone on the second call) — model that here since this
        # test uses a mock speaker_store.
        speaker_store = _make_speaker_store(speaker_id)
        speaker_store.remove.side_effect = [True, False]
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=speaker_store,
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )
        client = _make_client(monkeypatch, state)

        first = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        assert first.status_code == 200, first.text
        assert first.json()["erased_keys"] == [key]

        second = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["erased_keys"] == []
        assert body["removed_speaker"] is False

    def test_bookkeeping_entries_and_router_index_cleared(self, tmp_path, monkeypatch):
        """After forget: bookkeeping gone, entry payload gone, and a QueryRouter
        rebuilt on the same store indexes no keys for the speaker.

        The independent ``QueryRouter`` rebuild below proves the STORE state
        is correct but says nothing about whether the handler itself reloads
        the live ``_state["router"]`` — that used to be unverified here,
        which masked the (now-fixed) missing reload; the state's own router
        mock is asserted directly too.
        """
        from paramem.server.router import QueryRouter

        speaker_id = "speaker0"
        key = "graph1"

        real_store = MemoryStore(replay_enabled=True)
        real_store.put(
            "episodic",
            key,
            {"key": key, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=compute_simhash(key, "Alice", "lives_in", "Berlin"),
        )
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        loop = _make_loop_with_store(real_store)
        cfg = _make_config(tmp_path)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )
        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        assert resp.status_code == 200, resp.text

        assert real_store.bookkeeping_for_key(key) is None
        assert real_store.get(key) is None

        router = QueryRouter(cfg.adapter_dir, real_store)
        assert router._speaker_key_index.get(speaker_id, set()) == set()

        # The live state's own router mock was reloaded by the handler.
        state["router"].reload.assert_called_once_with()

    def test_promoted_keys_and_key_metadata_rewritten(self, tmp_path, monkeypatch):
        """loop.promoted_keys drops the forgotten key and key_metadata.json on
        disk is rewritten without it; a surviving key is untouched.

        Uses a real ``ConsolidationLoop`` stub (``object.__new__``) rather
        than ``_make_loop_with_store``'s ``MagicMock``: the handler's
        post-erase call now calls ``ConsolidationLoop.write_key_metadata``
        directly (the fold's own durable writer), and a ``MagicMock``'s
        ``write_key_metadata`` is itself a mock that never touches the store
        or the filesystem, so no file would be written at all — this test
        needs the real method to run.
        """
        from paramem.backup.encryption import read_maybe_encrypted
        from paramem.training.consolidation import ConsolidationLoop

        speaker_id = "speaker0"
        forgotten = "graph1"
        surviving = "graph2"

        real_store = MemoryStore(replay_enabled=True)
        real_store.put(
            "episodic",
            forgotten,
            {"key": forgotten, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=compute_simhash(forgotten, "Alice", "lives_in", "Berlin"),
        )
        real_store.set_bookkeeping(
            forgotten, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )
        real_store.put(
            "episodic",
            surviving,
            {"key": surviving, "subject": "Bob", "predicate": "has_job", "object": "Engineer"},
            simhash=compute_simhash(surviving, "Bob", "has_job", "Engineer"),
        )
        real_store.set_bookkeeping(
            surviving, speaker_id="speaker1", relation_type="factual", first_seen=""
        )

        cfg = _make_config(tmp_path)

        # Minimal real ConsolidationLoop stub — object.__new__ so no model
        # or GPU is needed.  Wires the same attributes _make_loop_with_store
        # pre-wires (store, model, ensure_adapters) so _reap_emptied_tiers'
        # dependencies resolve identically if this erase ever empties a
        # tier.  ``_key_metadata_path`` is threaded explicitly from
        # ``cfg.key_metadata_path`` — the same resolution every production
        # construction site performs via ``create_consolidation_loop`` —
        # since ``write_key_metadata`` has no config-fallback of its own.
        loop = object.__new__(ConsolidationLoop)
        loop.store = real_store
        loop.cycle_count = 0
        loop.model = MagicMock()
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)
        loop.promoted_keys = {forgotten, surviving}
        loop._key_metadata_path = cfg.key_metadata_path
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )
        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        assert resp.status_code == 200, resp.text

        assert forgotten not in loop.promoted_keys
        assert surviving in loop.promoted_keys

        raw = read_maybe_encrypted(cfg.key_metadata_path)
        metadata = json.loads(raw.decode("utf-8"))
        assert forgotten not in metadata["keys"]
        assert forgotten not in metadata["promoted_keys"]
        assert surviving in metadata["keys"]
        assert surviving in metadata["promoted_keys"]


# ---------------------------------------------------------------------------
# Simulate-mode graph.json content erasure
# ---------------------------------------------------------------------------


class TestGraphContentErasure:
    """The fact content in a tier's on-disk ``graph.json`` is erased, not
    just the registry pointer — the resurrection surface closed by
    ``erase_keys_from_graph_file``."""

    def test_main_tier_graph_edge_erased_survivor_kept(self, tmp_path, monkeypatch):
        """A main-tier graph.json: the erased key's edge is removed; the
        surviving edge's data (and its otherwise-isolated node) survive
        unchanged, and the file still exists afterward."""
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        cfg = _make_config(tmp_path)
        graph_path = cfg.adapter_dir / tier_name / "graph.json"
        g = nx.MultiDiGraph()
        g.add_edge(
            "Alice",
            "Berlin",
            **{_IK_KEY_ATTR: key_to_forget, "predicate": "lives_in", "speaker_id": speaker_id},
        )
        g.add_edge(
            "Bob",
            "Engineer",
            **{_IK_KEY_ATTR: key_to_keep, "predicate": "has_job", "speaker_id": "speaker1"},
        )
        save_memory_to_disk(g, graph_path)

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        assert graph_path.exists()

        g2 = load_memory_from_disk(graph_path)
        entries = {e["key"]: e for e in iter_entries(g2)}
        assert key_to_forget not in entries
        assert key_to_keep in entries
        assert entries[key_to_keep]["subject"] == "Bob"
        assert entries[key_to_keep]["object"] == "Engineer"
        assert entries[key_to_keep]["predicate"] == "has_job"
        assert entries[key_to_keep]["speaker_id"] == "speaker1"
        assert "Alice" not in g2
        assert "Berlin" not in g2

    def test_interim_slot_graph_edge_erased(self, tmp_path, monkeypatch):
        """An interim-slot graph.json (nested ``episodic/interim_<stamp>/``
        layout): the nested file is the one rewritten — no legacy top-level
        dir is minted, extending the existing interim layout assertions."""
        from paramem.memory.interim_adapter import (
            INTERIM_NAME_PREFIX,
            detect_legacy_adapter_layout,
        )

        stamp = "20260803T1200"
        tier_name = f"{INTERIM_NAME_PREFIX}{stamp}"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        adapter_dir = cfg.adapter_dir

        real_store = MemoryStore(replay_enabled=True)
        interim_reg = real_store.registry(tier_name)
        interim_reg.add(key_to_forget)
        interim_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget, speaker_id=speaker_id, relation_type="episodic", first_seen=""
        )

        interim_root = adapter_dir / "episodic" / f"interim_{stamp}"
        graph_path = interim_root / "graph.json"
        g = nx.MultiDiGraph()
        g.add_edge(
            "Alice",
            "Berlin",
            **{_IK_KEY_ATTR: key_to_forget, "predicate": "lives_in", "speaker_id": speaker_id},
        )
        g.add_edge(
            "Carl",
            "Chess",
            **{_IK_KEY_ATTR: key_to_keep, "predicate": "likes", "speaker_id": "speaker1"},
        )
        save_memory_to_disk(g, graph_path)

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        # No legacy top-level dir was minted.
        assert detect_legacy_adapter_layout(adapter_dir) == []
        assert not (adapter_dir / tier_name).exists()

        # The nested file is the one rewritten.
        assert graph_path.exists()
        g2 = load_memory_from_disk(graph_path)
        entries = {e["key"]: e for e in iter_entries(g2)}
        assert key_to_forget not in entries
        assert key_to_keep in entries


# ---------------------------------------------------------------------------
# Integration: the real handler's output fed to the real migrate() — the
# composition where the resurrection defect actually happened.
# ---------------------------------------------------------------------------


class TestForgetThenMigrateDoesNotResurrect:
    """Run the real ``/speaker/forget`` handler over a real on-disk simulate
    tier, then run ``active_store_migration.migrate()`` against that exact
    output directory with the GPU stack stubbed exactly as
    ``test_active_store_migration.py::TestMigrateTierSimulateToTrain
    .test_happy_path_orchestration`` does — nothing erased comes back.

    Other tests in this file and in ``test_active_store_migration.py`` pin
    the two halves — the graph erase and the migration filter — in
    isolation; this test pins the composition, which is where the
    resurrection defect actually happened.
    """

    def test_forget_then_migrate_does_not_resurrect_erased_key(self, tmp_path, monkeypatch):
        from paramem.server.active_store_migration import MigrationState, migrate

        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)
        ep_reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        cfg = _make_config(tmp_path)
        graph_path = cfg.adapter_dir / tier_name / "graph.json"
        g = nx.MultiDiGraph()
        g.add_edge(
            "Alice",
            "Berlin",
            **{_IK_KEY_ATTR: key_to_forget, "predicate": "lives_in", "speaker_id": speaker_id},
        )
        g.add_edge(
            "Bob",
            "Engineer",
            **{_IK_KEY_ATTR: key_to_keep, "predicate": "has_job", "speaker_id": "speaker1"},
        )
        save_memory_to_disk(g, graph_path)

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        # --- run the real /speaker/forget handler ---
        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        assert resp.status_code == 200, resp.text
        assert resp.json()["erased_keys"] == [key_to_forget]

        # --- run the real migrate() against the handler's own output dir,
        # with the GPU stack stubbed like test_happy_path_orchestration ---
        loop.episodic_config = MagicMock()
        loop.semantic_config = MagicMock()
        loop.procedural_config = MagicMock()
        loop.training_config = MagicMock(num_epochs=2)
        loop.config = MagicMock()
        loop.config.recall_sanity_threshold = 1.0
        loop.model = MagicMock()
        loop.model.peft_config = {}
        loop.tokenizer = MagicMock()
        loop.fingerprint_cache = None
        loop._train_tier_adapter.return_value = ({"aborted": False}, None)
        loop._run_recall_sanity_probe.return_value = 1.0

        def _fake_cache_entry(*, key, subject, predicate, object, speaker_id, **_kw):
            return {
                "key": key,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "speaker_id": speaker_id,
            }

        loop._cache_entry.side_effect = _fake_cache_entry

        slot_path = cfg.adapter_dir / tier_name / "20260430-000000"
        with (
            patch(
                "paramem.memory.entry.build_registry",
                return_value={key_to_keep: 0},
            ),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            migration_state = MigrationState.for_mode_switch(
                source_mode="simulate", target_mode="train"
            )
            migrate(loop, cfg, migration_state)

        assert loop._train_tier_adapter.call_count == 1
        trained_entries = loop._train_tier_adapter.call_args.args[0]
        trained_keys = {e["key"] for e in trained_entries}
        assert key_to_forget not in trained_keys, (
            f"erased key resurrected in migrated entries: {trained_keys}"
        )
        assert trained_keys == {key_to_keep}

        post_registry = real_store.registry(tier_name)
        assert not post_registry.knows(key_to_forget)

        on_disk_registry_path = cfg.adapter_dir / tier_name / "indexed_key_registry.json"
        on_disk = json.loads(on_disk_registry_path.read_bytes())
        assert key_to_forget not in on_disk.get("active_keys", [])


# ---------------------------------------------------------------------------
# Guard matrix — consolidating / bg-training / cloud-only / trial-active all
# refuse with 409 and never touch the store.  Mirrors
# tests/server/test_interim_discard.py::TestGuardMatrix.
# ---------------------------------------------------------------------------


class TestGuardMatrix:
    def _assert_refused_without_mutation(self, monkeypatch, state):
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, ["graph1"])
        state["consolidation_loop"] = loop
        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        return resp, loop

    def test_consolidating_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, consolidating=True)
        resp, loop = self._assert_refused_without_mutation(monkeypatch, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidating"
        loop.store.discard_keys.assert_not_called()

    def test_bg_training_refuses_409(self, tmp_path, monkeypatch):
        trainer = MagicMock()
        trainer.is_training = True
        state = _make_state(tmp_path, background_trainer=trainer)
        resp, loop = self._assert_refused_without_mutation(monkeypatch, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "training_active"
        loop.store.discard_keys.assert_not_called()

    def test_cloud_only_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="cloud-only")
        resp, loop = self._assert_refused_without_mutation(monkeypatch, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "cloud_only"
        loop.store.discard_keys.assert_not_called()

    def test_trial_active_refuses_409(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, migration={"state": "TRIAL"})
        resp, loop = self._assert_refused_without_mutation(monkeypatch, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"
        loop.store.discard_keys.assert_not_called()

    def test_base_swap_active_refuses_409(self, tmp_path, monkeypatch):
        """migration.base_swap_active=True refuses 409, same as the other
        four guard verdicts — closes the window where a base-swap
        orchestration's direct (non-mode-flipping) dispatch could win
        ``gpu_lock`` ahead of this door and release + recreate the
        ConsolidationLoop this door already captured a reference to.
        """
        state = _make_state(tmp_path, migration={"base_swap_active": True})
        resp, loop = self._assert_refused_without_mutation(monkeypatch, state)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "base_swap_active"
        loop.store.discard_keys.assert_not_called()


# ---------------------------------------------------------------------------
# Loop staleness across gpu_lock — a config-apply (or, before this fix, a
# base-swap orchestration racing ahead of the base_swap_active guard) that
# releases + recreates the ConsolidationLoop while this door holds gpu_lock
# must not leave the sync worker mutating the released pre-lock loop.
# ---------------------------------------------------------------------------


class TestLoopStalenessAcrossLock:
    def test_forget_sync_mutates_the_fresh_loop_not_the_pre_lock_capture(
        self, tmp_path, monkeypatch
    ):
        """Swap ``_state['consolidation_loop']`` to a fresh mock while the
        handler holds ``gpu_lock`` (simulating a config-apply that raced
        ahead of this door and won the lock first) — the executor-side
        mutation (``store.discard_keys``) must land on the FRESH loop, not
        the loop the handler captured before the lock, which proves
        ``_forget_sync`` re-resolves ``loop`` as its first statement rather
        than closing over the pre-lock capture.
        """
        import contextlib

        import paramem.server.gpu_lock as gpu_lock_module

        speaker_id = "speaker0"
        keys = ["graph1"]
        loop_a = _make_loop(speaker_id, keys)  # the handler's pre-lock capture
        loop_b = _make_loop(speaker_id, keys)  # swapped in while the lock is held

        state = _make_state(
            tmp_path,
            loop=loop_a,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        @contextlib.asynccontextmanager
        async def _swap_loop_gpu_lock():
            state["consolidation_loop"] = loop_b
            yield

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(gpu_lock_module, "gpu_lock", _swap_loop_gpu_lock)

        with patch("paramem.memory.persistence.save_registry"):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        loop_a.store.discard_keys.assert_not_called()
        loop_b.store.discard_keys.assert_called_once_with(sorted(keys), mode="erase")

    def test_forget_sync_mutates_the_fresh_config_not_the_pre_lock_capture(
        self, tmp_path, monkeypatch
    ):
        """Same race, but on ``_state['config']`` rather than the loop: a
        config-apply that wins ``gpu_lock`` first also replaces
        ``_state["config"]``.  ``_forget_sync``'s registry-write path
        (``adapter_slot_root_for_name(config.adapter_dir, tier_name)``) must
        resolve against the FRESH config's ``adapter_dir``, not the pre-lock
        one — proves the sync worker re-resolves ``config`` too, not just
        ``loop``.
        """
        import contextlib

        import paramem.server.gpu_lock as gpu_lock_module

        speaker_id = "speaker0"
        keys = ["graph1"]
        cfg_a = _make_config(tmp_path / "a")  # the handler's pre-lock capture
        cfg_b = _make_config(tmp_path / "b")  # swapped in while the lock is held
        loop_a = _make_loop(speaker_id, keys)
        loop_b = _make_loop(speaker_id, keys)

        state = _make_state(
            tmp_path,
            loop=loop_a,
            config=cfg_a,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        @contextlib.asynccontextmanager
        async def _swap_config_and_loop_gpu_lock():
            state["config"] = cfg_b
            state["consolidation_loop"] = loop_b
            yield

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(gpu_lock_module, "gpu_lock", _swap_config_and_loop_gpu_lock)

        with patch("paramem.memory.persistence.save_registry"):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        # The pre-lock loop/config pair is entirely untouched...
        loop_a._ep_registry.save.assert_not_called()
        # ...the fresh (swapped-in) loop's registry is written under the
        # fresh (swapped-in) config's adapter_dir.
        loop_b._ep_registry.save.assert_called_once()
        written_path = loop_b._ep_registry.save.call_args.args[0]
        assert cfg_b.adapter_dir in written_path.parents
        assert cfg_a.adapter_dir not in written_path.parents


# ---------------------------------------------------------------------------
# Reap: a tier this forget reduces to zero known keys is unmounted and
# deleted on the spot, instead of manifest re-stamped (owner decision,
# 2026-08-04).
# ---------------------------------------------------------------------------


class TestReapEmptiedTiers:
    def test_emptied_interim_slot_reaped(self, tmp_path, monkeypatch):
        """Sole key in an interim tier: tier gone from the store, its PEFT
        adapter deleted, its slot dir gone, its manifest_status row popped,
        and the response lists it."""
        stamp = "20260805T1200"
        tier_name = f"episodic_interim_{stamp}"
        key = "graph1"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        interim_reg = real_store.registry(tier_name)
        interim_reg.add(key)
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="episodic", first_seen=""
        )

        interim_root = cfg.adapter_dir / "episodic" / f"interim_{stamp}"
        interim_reg.save(interim_root / "indexed_key_registry.json")
        slot_dir = interim_root / "20260805-120000"
        slot_dir.mkdir(parents=True)
        write_manifest(
            slot_dir,
            _minimal_manifest_for_tier(
                tier_name, hashlib.sha256(interim_reg.save_bytes()).hexdigest(), key_count=1
            ),
        )

        model = _make_peft_model("episodic", "semantic", "procedural", tier_name)
        loop = _make_loop_with_store(real_store)
        loop.model = model
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )
        state["adapter_manifest_status"] = {
            tier_name: {"status": "healthy"},
            "episodic": {"status": "healthy"},
        }

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reaped_tiers"] == [tier_name]
        assert body["unloaded_adapters"] == [tier_name]
        assert body["removed_dirs"] == [f"interim_{stamp}"]

        assert tier_name not in real_store.tiers_with_registry()
        assert tier_name not in model.peft_config
        assert not interim_root.exists()
        assert tier_name not in state["adapter_manifest_status"]
        assert "episodic" in state["adapter_manifest_status"]

    def test_sibling_interim_slot_with_other_speaker_keys_survives(self, tmp_path, monkeypatch):
        """Two interim tiers, each holding a different speaker's key: only
        the forgotten speaker's tier is reaped; the sibling is untouched in
        the store, on disk, and in PEFT."""
        stamp_a = "20260805T1200"
        stamp_b = "20260805T1300"
        tier_a = f"episodic_interim_{stamp_a}"
        tier_b = f"episodic_interim_{stamp_b}"
        speaker_id = "speaker0"
        other_speaker = "speaker1"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        reg_a = real_store.registry(tier_a)
        reg_a.add("graph1")
        real_store.set_bookkeeping(
            "graph1", speaker_id=speaker_id, relation_type="episodic", first_seen=""
        )
        reg_b = real_store.registry(tier_b)
        reg_b.add("graph2")
        real_store.set_bookkeeping(
            "graph2", speaker_id=other_speaker, relation_type="episodic", first_seen=""
        )

        root_a = cfg.adapter_dir / "episodic" / f"interim_{stamp_a}"
        reg_a.save(root_a / "indexed_key_registry.json")
        root_b = cfg.adapter_dir / "episodic" / f"interim_{stamp_b}"
        reg_b.save(root_b / "indexed_key_registry.json")

        model = _make_peft_model("episodic", tier_a, tier_b)
        loop = _make_loop_with_store(real_store)
        loop.model = model
        loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reaped_tiers"] == [tier_a]

        assert tier_a not in real_store.tiers_with_registry()
        assert tier_b in real_store.tiers_with_registry()
        assert not root_a.exists()
        assert root_b.exists()
        assert tier_a not in model.peft_config
        assert tier_b in model.peft_config

    def test_emptied_main_tier_reaped_to_pristine_shape(self, tmp_path, monkeypatch):
        """Sole key in a main tier: the tier root's registry + weight slot
        are gone, but a sibling ``interim_*`` child under the same root
        survives untouched — reap_tier_artifacts never removes an
        ``interim_*`` child, and the root itself is left standing because
        the child keeps it non-empty."""
        tier_name = "episodic"
        key = "graph1"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        reg = real_store.registry(tier_name)
        reg.add(key)
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        tier_root = cfg.adapter_dir / tier_name
        reg.save(tier_root / "indexed_key_registry.json")
        slot_dir = tier_root / "20260805-120000"
        slot_dir.mkdir(parents=True)
        write_manifest(
            slot_dir,
            _minimal_manifest_for_tier(
                tier_name, hashlib.sha256(reg.save_bytes()).hexdigest(), key_count=1
            ),
        )
        # A sibling live interim tier nested under the same root — must survive.
        interim_child = tier_root / "interim_20260805T1400"
        interim_child.mkdir(parents=True)
        (interim_child / "graph.json").write_text("{}")

        loop = _make_loop_with_store(real_store)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reaped_tiers"] == [tier_name]

        assert not (tier_root / "indexed_key_registry.json").exists()
        assert not slot_dir.exists()
        assert tier_root.exists(), "root must survive — the interim child keeps it non-empty"
        assert interim_child.exists()
        assert (interim_child / "graph.json").exists()
        loop.ensure_adapters.assert_called_once()

    def test_total_wipe_leaves_serviceable_model(self, tmp_path, monkeypatch):
        """All three main tiers emptied at once: every adapter is detached,
        ensure_adapters is called exactly once to rebuild a serviceable
        model, that fresh model is re-published to _state["model"], and its
        active adapter is switched to episodic."""
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        for tier, key in (
            ("episodic", "graph1"),
            ("semantic", "graph2"),
            ("procedural", "graph3"),
        ):
            reg = real_store.registry(tier)
            reg.add(key)
            real_store.set_bookkeeping(
                key, speaker_id=speaker_id, relation_type="factual", first_seen=""
            )
            reg.save(cfg.adapter_dir / tier / "indexed_key_registry.json")

        original_model = _make_peft_model("episodic", "semantic", "procedural")
        fresh_model = _make_peft_model("episodic")

        loop = _make_loop_with_store(real_store)
        loop.model = original_model
        loop.ensure_adapters = MagicMock(return_value=fresh_model)

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert sorted(body["reaped_tiers"]) == ["episodic", "procedural", "semantic"]
        assert sorted(body["unloaded_adapters"]) == ["episodic", "procedural", "semantic"]

        loop.ensure_adapters.assert_called_once()
        assert state["model"] is fresh_model
        fresh_model.set_adapter.assert_called_once_with("episodic")


# ---------------------------------------------------------------------------
# A surviving tier keeps today's registry-level posture — not reaped.
# ---------------------------------------------------------------------------


class TestPartialForgetKeepsPosture:
    def test_survivor_tier_not_reaped(self, tmp_path, monkeypatch):
        tier_name = "episodic"
        key_to_forget = "graph1"
        key_to_keep = "graph2"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        reg = real_store.registry(tier_name)
        reg.add(key_to_forget)
        reg.add(key_to_keep)
        real_store.set_bookkeeping(
            key_to_forget, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )
        tier_root = cfg.adapter_dir / tier_name
        reg.save(tier_root / "indexed_key_registry.json")

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reaped_tiers"] == []
        assert body["unloaded_adapters"] == []
        assert body["removed_dirs"] == []
        assert tier_root.exists()
        assert (tier_root / "indexed_key_registry.json").exists()
        assert tier_name in real_store.tiers_with_registry()


# ---------------------------------------------------------------------------
# Router reload
# ---------------------------------------------------------------------------


class TestRouterReloadedOnce:
    def test_router_reload_called_once(self, tmp_path, monkeypatch):
        speaker_id = "speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        state["router"].reload.assert_called_once_with()


# ---------------------------------------------------------------------------
# Ring-lifecycle incidents — resolved only when this forget's reap empties
# the interim ring entirely.
# ---------------------------------------------------------------------------


class TestRingIncidentsOnReap:
    def _record_ring_incidents(self, state_dir):
        for t in (
            "full_consolidation_overdue",
            "interim_cap_reached",
            "interim_overflow_pending",
            "vram_exhausted",
        ):
            record_incident(
                state_dir, type=t, key="k1", severity="warning", summary="test", detail={}
            )

    def test_incidents_resolved_when_ring_fully_emptied(self, tmp_path, monkeypatch):
        stamp = "20260805T1200"
        tier_name = f"episodic_interim_{stamp}"
        key = "graph1"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        state_dir = cfg.paths.data / "state"
        self._record_ring_incidents(state_dir)

        real_store = MemoryStore(replay_enabled=True)
        reg = real_store.registry(tier_name)
        reg.add(key)
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="episodic", first_seen=""
        )
        interim_root = cfg.adapter_dir / "episodic" / f"interim_{stamp}"
        reg.save(interim_root / "indexed_key_registry.json")

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        incidents = {i.type: i for i in read_incidents(state_dir)}
        for t in ("full_consolidation_overdue", "interim_cap_reached", "interim_overflow_pending"):
            assert incidents[t].status == "resolved"
        assert incidents["vram_exhausted"].status == "active"

    def test_incidents_untouched_when_sibling_interim_survives(self, tmp_path, monkeypatch):
        stamp_a = "20260805T1200"
        stamp_b = "20260805T1300"
        tier_a = f"episodic_interim_{stamp_a}"
        tier_b = f"episodic_interim_{stamp_b}"
        speaker_id = "speaker0"
        other_speaker = "speaker1"

        cfg = _make_config(tmp_path)
        state_dir = cfg.paths.data / "state"
        self._record_ring_incidents(state_dir)

        real_store = MemoryStore(replay_enabled=True)
        reg_a = real_store.registry(tier_a)
        reg_a.add("graph1")
        real_store.set_bookkeeping(
            "graph1", speaker_id=speaker_id, relation_type="episodic", first_seen=""
        )
        reg_b = real_store.registry(tier_b)
        reg_b.add("graph2")
        real_store.set_bookkeeping(
            "graph2", speaker_id=other_speaker, relation_type="episodic", first_seen=""
        )
        root_a = cfg.adapter_dir / "episodic" / f"interim_{stamp_a}"
        root_b = cfg.adapter_dir / "episodic" / f"interim_{stamp_b}"
        reg_a.save(root_a / "indexed_key_registry.json")
        reg_b.save(root_b / "indexed_key_registry.json")

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        incidents = {i.type: i for i in read_incidents(state_dir)}
        for t in ("full_consolidation_overdue", "interim_cap_reached", "interim_overflow_pending"):
            assert incidents[t].status == "active"


# ---------------------------------------------------------------------------
# _state["consolidating"] lifecycle
# ---------------------------------------------------------------------------


class TestConsolidatingFlagLifecycle:
    def test_flag_true_during_mutation_false_after(self, tmp_path, monkeypatch):
        speaker_id = "speaker0"
        key = "graph1"
        real_store = MemoryStore(replay_enabled=True)
        real_store.put(
            "episodic",
            key,
            {"key": key, "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            simhash=compute_simhash(key, "Alice", "lives_in", "Berlin"),
        )
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )

        loop = _make_loop_with_store(real_store)
        cfg = _make_config(tmp_path)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        observed: list[bool] = []
        orig_discard_keys = real_store.discard_keys

        def _spy(*args, **kwargs):
            observed.append(state["consolidating"])
            return orig_discard_keys(*args, **kwargs)

        monkeypatch.setattr(real_store, "discard_keys", _spy)

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        assert observed == [True]
        assert state["consolidating"] is False


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestReapFailure:
    def test_reap_failure_returns_500_and_clears_flag(self, tmp_path, monkeypatch):
        tier_name = "episodic"
        key = "graph1"
        speaker_id = "speaker0"

        cfg = _make_config(tmp_path)
        real_store = MemoryStore(replay_enabled=True)
        reg = real_store.registry(tier_name)
        reg.add(key)
        real_store.set_bookkeeping(
            key, speaker_id=speaker_id, relation_type="factual", first_seen=""
        )
        reg.save(cfg.adapter_dir / tier_name / "indexed_key_registry.json")

        loop = _make_loop_with_store(real_store)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("reap failed")

        monkeypatch.setattr("paramem.memory.persistence.reap_tier_artifacts", _boom)

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 500
        assert state["consolidating"] is False
