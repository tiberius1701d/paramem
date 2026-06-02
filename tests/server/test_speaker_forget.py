"""Tests for ``POST /speaker/forget``.

Mocked — no GPU, no real model.  Mirrors the mocking style of
``tests/server/test_gates.py`` and ``tests/server/test_integrity_endpoint.py``.

Coverage
--------
- ``mark_stale`` called with exactly the keys ``keys_for_speaker`` returns for
  the speaker: ``KeyRegistry.remove`` called for each key; registry saved.
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

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module
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
    """Build a MagicMock ConsolidationLoop whose merger.graph returns *keys* for *speaker_id*.

    The store exposes two main tiers (``episodic`` and ``semantic``) whose
    ``KeyRegistry`` objects contain the supplied keys so ``remove`` and ``save``
    can be verified.  The simhash dicts include the keys so the simhash
    clean-up branch is exercised.
    """
    loop = MagicMock()

    # Merger graph — keys_for_speaker is patched at call site, so any graph is fine.
    loop.merger.graph = MagicMock()

    # Per-tier KeyRegistry mocks.
    ep_registry = MagicMock(spec=KeyRegistry)
    ep_registry.__contains__ = MagicMock(side_effect=lambda k: k in keys)
    sem_registry = MagicMock(spec=KeyRegistry)
    sem_registry.__contains__ = MagicMock(side_effect=lambda _: False)

    # Store: tiers_with_registry returns episodic + semantic.
    loop.store.tiers_with_registry.return_value = ["episodic", "semantic"]
    loop.store.registry.side_effect = lambda t: ep_registry if t == "episodic" else sem_registry

    # Simhash dicts: episodic holds the keys; semantic is empty.
    ep_simhash = {k: i + 1 for i, k in enumerate(keys)}
    sem_simhash: dict = {}
    loop.store.simhashes_in_tier.side_effect = lambda t: (
        ep_simhash if t == "episodic" else sem_simhash
    )

    loop._ep_registry = ep_registry
    loop._sem_registry = sem_registry
    loop._ep_simhash = ep_simhash
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
    """Keys returned by keys_for_speaker are removed from KeyRegistry and registry saved."""

    def test_keys_removed_from_registry_and_saved(self, tmp_path, monkeypatch):
        """KeyRegistry.remove called for each key; registry.save called with tier path."""
        speaker_id = "Speaker0"
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

        with (
            patch(
                "paramem.memory.persistence.keys_for_speaker", return_value=set(keys)
            ) as mock_kfs,
            patch("paramem.memory.persistence.save_registry"),
        ):
            client = _make_client(monkeypatch, state)
            resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Response lists stale keys (sorted).
        assert sorted(body["stale_keys"]) == sorted(keys)

        # keys_for_speaker was called with the graph and speaker_id.
        mock_kfs.assert_called_once_with(loop.merger.graph, speaker_id)

        # remove called for each key on the episodic registry (which contains them).
        ep_reg = loop._ep_registry
        remove_calls = {c.args[0] for c in ep_reg.remove.call_args_list}
        assert remove_calls == set(keys)

        # save called with the episodic adapter path.
        expected_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        ep_reg.save.assert_called_once_with(expected_path)

        # Semantic registry — no keys → remove and save not called.
        loop._sem_registry.remove.assert_not_called()
        loop._sem_registry.save.assert_not_called()

    def test_simhash_entries_removed_and_persisted(self, tmp_path, monkeypatch):
        """Simhash entries for the speaker's keys are removed and re-saved."""
        speaker_id = "Speaker0"
        keys = ["graph2"]
        loop = _make_loop(speaker_id, keys)
        cfg = _make_config(tmp_path)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
            config=cfg,
        )

        saved_simhash_calls = []
        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set(keys)):
            with patch(
                "paramem.memory.persistence.save_registry",
                side_effect=lambda reg, path: saved_simhash_calls.append((dict(reg), path)),
            ):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200, resp.text

        # The episodic simhash registry was saved after the key was removed.
        ep_saves = [r for r, p in saved_simhash_calls if "episodic" in str(p)]
        assert ep_saves, "No episodic simhash save recorded"
        # Key must be absent from the saved dict.
        assert "graph2" not in ep_saves[0]

    def test_no_keys_no_registry_mutations(self, tmp_path, monkeypatch):
        """When keys_for_speaker returns empty set, no registry mutations occur."""
        speaker_id = "Speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry") as mock_save:
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["stale_keys"] == []
        # No registry saves because there are no keys to process.
        mock_save.assert_not_called()
        loop._ep_registry.remove.assert_not_called()


# ---------------------------------------------------------------------------
# Speaker profile removal
# ---------------------------------------------------------------------------


class TestSpeakerProfileRemoval:
    """speaker_store.remove called; response removed_speaker reflects its bool."""

    def test_profile_found_and_removed(self, tmp_path, monkeypatch):
        """SpeakerStore.remove returns True → removed_speaker is True."""
        speaker_id = "Speaker1"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id, returns=True),
            buffer=_make_buffer(speaker_id, []),
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed_speaker"] is True
        state["speaker_store"].remove.assert_called_once_with(speaker_id)

    def test_profile_not_found(self, tmp_path, monkeypatch):
        """SpeakerStore.remove returns False → removed_speaker is False."""
        speaker_id = "Speaker99"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id, returns=False),
            buffer=_make_buffer(speaker_id, []),
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed_speaker"] is False

    def test_no_speaker_store(self, tmp_path, monkeypatch):
        """When speaker_store is None (voice disabled), removed_speaker is False."""
        speaker_id = "Speaker0"
        loop = _make_loop(speaker_id, [])
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=None,
            buffer=_make_buffer(speaker_id, []),
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
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
        speaker_id = "Speaker0"
        conv_ids = ["conv-a", "conv-b"]
        loop = _make_loop(speaker_id, [])
        buffer = _make_buffer(speaker_id, conv_ids)
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=buffer,
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert sorted(body["discarded_sessions"]) == sorted(conv_ids)
        buffer.discard_sessions.assert_called_once_with(conv_ids)

    def test_no_pending_sessions(self, tmp_path, monkeypatch):
        """When no pending sessions exist for the speaker, discarded_sessions is empty."""
        speaker_id = "Speaker0"
        loop = _make_loop(speaker_id, [])
        buffer = MagicMock()
        buffer._sessions = {"other-conv": {"speaker_id": "Speaker1"}}
        state = _make_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=buffer,
        )

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
                client = _make_client(monkeypatch, state)
                resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["discarded_sessions"] == []
        buffer.discard_sessions.assert_not_called()

    def test_sessions_with_mixed_speakers_only_discards_target(self, tmp_path, monkeypatch):
        """Only sessions whose speaker_id matches the request are discarded."""
        speaker_id = "Speaker0"
        other_id = "Speaker1"
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

        with patch("paramem.memory.persistence.keys_for_speaker", return_value=set()):
            with patch("paramem.memory.persistence.save_registry"):
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
            speaker_store=_make_speaker_store("Speaker0"),
            buffer=MagicMock(),
        )

        client = _make_client(monkeypatch, state)
        resp = client.post("/speaker/forget", json={"speaker_id": "Speaker0"})

        assert resp.status_code == 503
        detail = resp.json().get("detail", {})
        assert detail.get("status") == "not_ready"

    def test_unsupported_strategy_returns_400(self, tmp_path, monkeypatch):
        """Requesting an unsupported strategy returns 400."""
        speaker_id = "Speaker0"
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
