"""Tests for ``POST /speaker/forget``.

Mocked — no GPU, no real model.  Mirrors the mocking style of
``tests/server/test_gates.py`` and ``tests/server/test_integrity_endpoint.py``.

Coverage
--------
- ``store.discard_keys(keys, mode="erase")`` called with exactly the keys
  resolved via ``store.iter_bookkeeping()`` for the speaker: hard-removes
  keys from the registry (GONE from both active and stale); registry saved
  for affected tiers; simhash saved for affected main tiers.
- Speaker profile removed: ``speaker_store.remove`` called; response reflects
  the bool return.
- Pending sessions for the speaker discarded: ``discard_sessions`` called;
  response lists them.
- ``consolidation_loop is None`` → 503 with ``"not_ready"`` status detail.
- Unsupported ``strategy`` → 400.
- Auth: ``/speaker/forget`` carries the ``require_admin`` dependency in the
  real app route table.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

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
from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> MagicMock:
    """Minimal config mock with adapter_dir under tmp_path."""
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    return cfg


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
    """
    loop = MagicMock()

    # W2: iter_bookkeeping returns bookkeeping records keyed by speaker_id.
    # The handler iterates all records and filters by record.get("speaker_id").
    bk_records = [(k, {"speaker_id": speaker_id, "relation_type": "episodic"}) for k in keys]
    loop.store.iter_bookkeeping.return_value = iter(bk_records)

    # Per-tier KeyRegistry mocks.
    ep_registry = MagicMock(spec=KeyRegistry)
    ep_registry.__contains__ = MagicMock(side_effect=lambda k: k in keys)
    ep_registry.is_stale = MagicMock(return_value=False)
    ep_registry.knows = MagicMock(side_effect=lambda k: k in keys)
    # save_bytes must return valid bytes so the handler can compute sha256.
    # Distinct pre/post values are not needed here because discard_keys is mocked
    # (it does not mutate the mock registry); any stable bytes value is sufficient.
    ep_registry.save_bytes.return_value = (
        b'{"active_keys": [], "fidelity_history": {}, "health": null, "stale": {}}'
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
) -> dict:
    """Build a minimal _state dict for endpoint tests."""
    return {
        "config": config or _make_config(tmp_path),
        "consolidation_loop": loop,
        "speaker_store": speaker_store,
        "session_buffer": buffer or MagicMock(),
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
        assert sorted(body["stale_keys"]) == sorted(keys)

        # W2: iter_bookkeeping was called (not keys_for_speaker which used merger.graph).
        loop.store.iter_bookkeeping.assert_called_once_with()

        # store.discard_keys must be called with mode="erase" (hard erasure, not soft-stale).
        # Verify the helper was called; the erase-vs-stale distinction is tested in
        # the store unit tests (TestDiscardKeys in test_memory_persistence/store tests).
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
        # W2: seed bookkeeping so iter_bookkeeping() resolves the key for this speaker.
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
        assert body["stale_keys"] == []
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
    """consolidation_loop is None → 503; unsupported strategy → 400."""

    def test_consolidation_loop_none_returns_503(self, tmp_path, monkeypatch):
        """When consolidation_loop is None, endpoint returns 503 with not_ready detail."""
        state = _make_state(
            tmp_path,
            loop=None,
            speaker_store=_make_speaker_store("speaker0"),
            buffer=MagicMock(),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": "speaker0"})

        assert resp.status_code == 503
        detail = resp.json().get("detail", {})
        assert detail.get("status") == "not_ready"

    def test_unsupported_strategy_returns_400(self, tmp_path, monkeypatch):
        """Requesting an unsupported strategy returns 400."""
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
            json={"speaker_id": speaker_id, "strategy": "discard_interim"},
        )

        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        assert detail.get("status") == "unsupported_strategy"


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
        loop = MagicMock()
        loop.store = real_store

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

        # W2: seed bookkeeping so iter_bookkeeping() resolves key_to_forget for speaker_id.
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
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path / "adapters"

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

    def test_no_slot_logs_error_without_failing(self, tmp_path, monkeypatch, caplog):
        """When no slot matches the pre-erase hash, an ERROR is logged but the request succeeds.

        This covers the already-orphaned slot case (e.g. after a prior crash between
        registry.save and manifest re-stamp).  The /forget must not fail — partial
        re-stamp state is recoverable via a full consolidation fold.
        """
        tier_name = "episodic"
        key_to_forget = "graph1"
        speaker_id = "speaker0"

        # Build a real KeyRegistry with the key.
        real_store = MemoryStore(replay_enabled=True)
        ep_reg = real_store.registry(tier_name)
        ep_reg.add(key_to_forget)

        # W2: seed bookkeeping so iter_bookkeeping() resolves key_to_forget for speaker_id.
        real_store.set_bookkeeping(
            key_to_forget,
            speaker_id=speaker_id,
            relation_type="factual",
            first_seen="",
        )

        # Write a slot with a DIFFERENT registry_sha256 so find_live_slot won't match H_old.
        slot_dir = tmp_path / "adapters" / tier_name / "20260612-000000"
        slot_dir.mkdir(parents=True)
        mismatched_manifest = _minimal_manifest_for_tier(tier_name, "deadbeef" * 8, key_count=1)
        write_manifest(slot_dir, mismatched_manifest)

        cfg = MagicMock()
        cfg.adapter_dir = tmp_path / "adapters"

        loop = MagicMock()
        loop.store = real_store

        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        # Explicitly wire caplog's handler onto the named logger so TestClient's
        # ASGI scope emits records into caplog.  Pattern from test_auto_reclaim_live_reload.py.
        named = logging.getLogger("paramem.server.app")
        orig_propagate = named.propagate
        named.propagate = True
        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        named.addHandler(caplog.handler)
        try:
            with patch("paramem.memory.persistence.save_registry"):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})
        finally:
            named.removeHandler(caplog.handler)
            named.propagate = orig_propagate

        assert resp.status_code == 200, resp.text
        # An ERROR must be logged about the missing slot.
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("slot already orphaned" in msg for msg in error_messages), (
            f"Expected orphaned-slot ERROR in logs, got: {error_messages}"
        )
