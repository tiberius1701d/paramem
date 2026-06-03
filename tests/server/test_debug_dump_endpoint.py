"""Tests for the GET /debug/dump endpoint.

Covers:
- 403 when ``config.debug=False`` (gating contract).
- 503 when ``memory_store`` is not constructed yet.
- Empty store → returns empty list, total=0 (correct read, not an error).
- Happy path: every (tier, key, entry) from iter_entries() flows into the response.
- Tier counts aggregate correctly.

Tests use FastAPI TestClient with monkeypatched ``_state``; no live server, no GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import paramem.server.app as app_module


def _make_config(tmp_path: Path, debug: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    cfg.paths.data = tmp_path / "data"
    return cfg


class _FakeStore:
    def __init__(self, items: list[tuple[str, str, dict]], bookkeeping: dict | None = None):
        self._items = items
        self._bookkeeping = bookkeeping or {}

    def iter_entries(self):
        yield from self._items

    def bookkeeping_count(self) -> int:
        return len(self._bookkeeping)


def _make_state(tmp_path: Path, *, debug: bool = True, store_items=None) -> dict:
    state = {"config": _make_config(tmp_path, debug=debug)}
    if store_items is not None:
        state["memory_store"] = _FakeStore(store_items)
    return state


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


class TestDebugDumpGating:
    def test_debug_false_returns_403(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, debug=False, store_items=[])
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 403
        assert resp.json()["status"] == "forbidden_not_debug"

    def test_no_memory_store_returns_503(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, store_items=None)  # store_items=None → key absent
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


class TestDebugDumpHappyPath:
    def test_empty_store_returns_empty_list_200(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, store_items=[])
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"entries": [], "total": 0, "tiers": {}, "bookkeeping_total": 0}

    def test_cache_off_entries_empty_bookkeeping_nonzero(self, tmp_path, monkeypatch):
        """Under preload_cache=False: entries is empty, bookkeeping_total is N."""
        # Bookkeeping present but no content entries (cache-off scenario).
        fake_bk = {
            "k1": {"speaker_id": "alice", "first_seen_cycle": 1},
            "k2": {"speaker_id": "alice", "first_seen_cycle": 2},
        }
        state = _make_state(tmp_path, store_items=[])
        state["memory_store"] = _FakeStore([], bookkeeping=fake_bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["bookkeeping_total"] == 2

    def test_dump_flattens_tier_key_entry(self, tmp_path, monkeypatch):
        items = [
            (
                "episodic",
                "graph1",
                {"subject": "Tobias", "predicate": "lives_in", "object": "Berlin"},
            ),
            (
                "episodic",
                "graph2",
                {"subject": "Tobias", "predicate": "works_at", "object": "Anthropic"},
            ),
            (
                "procedural",
                "proc1",
                {"subject": "Tobias", "predicate": "prefers", "object": "concise answers"},
            ),
        ]
        state = _make_state(tmp_path, store_items=items)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 3
        assert body["tiers"] == {"episodic": 2, "procedural": 1}
        assert len(body["entries"]) == 3
        # Each row carries tier + key + the entry payload, flattened.
        first = body["entries"][0]
        assert first["tier"] == "episodic"
        assert first["key"] == "graph1"
        assert first["subject"] == "Tobias"
        assert first["object"] == "Berlin"
